# ============================================================================
# DNS — Route 53 Hosted Zone + A Records
#
# After `terraform apply`, copy the nameservers from output and update
# your domain registrar's NS records to point to them.
# ============================================================================

resource "aws_route53_zone" "main" {
  name = var.domain

  tags = { Name = "traderbot-zone" }
}

# Root domain → Elastic IP
resource "aws_route53_record" "root" {
  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain
  type    = "A"
  ttl     = 300
  records = [aws_eip.app.public_ip]
}

# www → Elastic IP (Caddy handles redirect to root)
resource "aws_route53_record" "www" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.${var.domain}"
  type    = "A"
  ttl     = 300
  records = [aws_eip.app.public_ip]
}
