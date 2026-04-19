# ============================================================================
# Frontend SPA — S3 + CloudFront + OAC
#
# Architectural Intent:
# - The React SPA ships as static files to S3. CloudFront fronts the
#   bucket with Origin Access Control (OAC), so the bucket itself is
#   private; only CloudFront can read it.
# - CloudFront routes /api/* and /ws to the ALB so users hit a single
#   origin and don't need CORS.
# - SPA fallback: 404 from the bucket is rewritten to /index.html so
#   client-side routing works for deep links.
# ============================================================================

# ── S3 bucket for SPA assets ───────────────────────────────────────────────
resource "aws_s3_bucket" "spa" {
  bucket = "${local.name_prefix}-spa-${data.aws_caller_identity.current.account_id}"
  tags   = { Name = "${local.name_prefix}-spa" }
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket_public_access_block" "spa" {
  bucket = aws_s3_bucket.spa.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "spa" {
  bucket = aws_s3_bucket.spa.id
  versioning_configuration {
    status = "Enabled" # lets CI roll back a bad SPA deploy
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "spa" {
  bucket = aws_s3_bucket.spa.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ── Bucket policy grants read-only to CloudFront via OAC ───────────────────
data "aws_iam_policy_document" "spa_bucket" {
  statement {
    effect  = "Allow"
    actions = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.spa.arn}/*"]

    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.spa.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "spa" {
  bucket = aws_s3_bucket.spa.id
  policy = data.aws_iam_policy_document.spa_bucket.json
}

# ── CloudFront distribution ────────────────────────────────────────────────
resource "aws_cloudfront_origin_access_control" "spa" {
  name                              = "${local.name_prefix}-spa-oac"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# Separate us-east-1 ACM cert for CloudFront (required).
resource "aws_acm_certificate" "spa" {
  provider          = aws.us_east_1
  domain_name       = var.domain
  validation_method = "DNS"

  subject_alternative_names = ["www.${var.domain}"]

  lifecycle {
    create_before_destroy = true
  }

  tags = { Name = "${local.name_prefix}-spa-cert" }
}

resource "aws_route53_record" "spa_cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.spa.domain_validation_options : dvo.domain_name => {
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

resource "aws_acm_certificate_validation" "spa" {
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.spa.arn
  validation_record_fqdns = [for r in aws_route53_record.spa_cert_validation : r.fqdn]
}

resource "aws_cloudfront_distribution" "spa" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "TraderBot SPA + API routing"
  default_root_object = "index.html"
  price_class         = "PriceClass_100" # US, Canada, Europe; cheapest tier
  aliases             = [var.domain, "www.${var.domain}"]

  # ── S3 origin for the SPA ─────────────────────────────────────────────
  origin {
    domain_name              = aws_s3_bucket.spa.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.spa.id
    origin_id                = "spa-s3"
  }

  # ── ALB origin for /api/* and /ws ─────────────────────────────────────
  origin {
    domain_name = aws_lb.api.dns_name
    origin_id   = "api-alb"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  # Default behaviour → SPA.
  default_cache_behavior {
    target_origin_id       = "spa-s3"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    # Managed cache policy — CachingOptimized.
    cache_policy_id = "658327ea-f89d-4fab-a63d-7e88639e58f6"
  }

  # /api/* → ALB, never cache API responses.
  ordered_cache_behavior {
    path_pattern           = "/api/*"
    target_origin_id       = "api-alb"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    # Managed CachingDisabled + AllViewer origin request policy.
    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
    origin_request_policy_id = "216adef6-5c7f-47e4-b989-5492eafa07d3"
  }

  # /ws → ALB for WebSocket upgrade.
  ordered_cache_behavior {
    path_pattern           = "/ws"
    target_origin_id       = "api-alb"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]
    compress               = false

    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
    origin_request_policy_id = "216adef6-5c7f-47e4-b989-5492eafa07d3"
  }

  # SPA fallback: React Router deep links must return index.html.
  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  custom_error_response {
    error_code            = 404
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.spa.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = { Name = "${local.name_prefix}-spa-cf" }
}
