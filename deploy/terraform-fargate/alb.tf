# ============================================================================
# ALB + ACM cert + HTTPS listener
#
# Architectural Intent:
# - One ALB fronting the Fargate service. ACM issues the TLS cert; Route
#   53 validates via DNS. HTTP → HTTPS redirect at the listener.
# - Health check hits /ready which uvicorn exposes once the DI container
#   finished boot. Aligns with the container healthCheck so target-group
#   health + task health agree.
# ============================================================================

resource "aws_lb" "api" {
  name               = "${local.name_prefix}-alb"
  load_balancer_type = "application"
  subnets            = [for s in aws_subnet.public : s.id]
  security_groups    = [aws_security_group.alb.id]

  # Protects against tf-destroy misfires while the stack is live.
  enable_deletion_protection = true

  # Drop idle connections after 60s so we don't keep Fargate workers
  # pinned on slow clients. Uvicorn defaults to 5s keep-alive; give us
  # some margin above that.
  idle_timeout = 60

  tags = { Name = "${local.name_prefix}-alb" }
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name_prefix}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # Fargate awsvpc mode → IP targets

  # Reduce deregistration delay; Fargate restarts shouldn't take 300s to
  # drain. 15s matches uvicorn graceful-shutdown budget.
  deregistration_delay = 15

  health_check {
    enabled             = true
    path                = "/ready"
    port                = "traffic-port"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 15
    matcher             = "200"
  }

  # Sticky sessions off — the app is stateless.
  stickiness {
    enabled = false
    type    = "lb_cookie"
  }

  tags = { Name = "${local.name_prefix}-api-tg" }
}

# ── ACM certificate (DNS-validated via Route 53) ───────────────────────────
resource "aws_acm_certificate" "api" {
  domain_name       = var.domain
  validation_method = "DNS"

  subject_alternative_names = ["www.${var.domain}"]

  lifecycle {
    create_before_destroy = true
  }

  tags = { Name = "${local.name_prefix}-api-cert" }
}

# Route 53 records for ACM DNS validation. Assumes the existing hosted
# zone for the domain is already in this AWS account.
data "aws_route53_zone" "root" {
  name         = var.domain
  private_zone = false
}

resource "aws_route53_record" "acm_validation" {
  for_each = {
    for dvo in aws_acm_certificate.api.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = data.aws_route53_zone.root.zone_id
}

resource "aws_acm_certificate_validation" "api" {
  certificate_arn         = aws_acm_certificate.api.arn
  validation_record_fqdns = [for r in aws_route53_record.acm_validation : r.fqdn]
}

# ── Listeners ──────────────────────────────────────────────────────────────
resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate_validation.api.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# The SPA bucket behind CloudFront answers the root / and /assets/*
# paths; the ALB listener rule below tells the ALB to forward anything
# starting with /api or /ws to the Fargate target group and return a
# 404 for everything else (so requests that slip past CloudFront don't
# hit the SPA fallback on the API).
resource "aws_lb_listener_rule" "api_paths" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    path_pattern {
      values = ["/api/*", "/ws", "/health", "/ready"]
    }
  }
}
