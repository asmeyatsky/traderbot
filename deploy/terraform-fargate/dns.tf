# ============================================================================
# Route 53 records
#
# Architectural Intent:
# - The apex (traderbotapp.com) and www resolve to CloudFront. CloudFront
#   handles SPA + /api/* + /ws, so clients only see one origin and CORS
#   goes away.
# - CloudFront itself routes /api/* and /ws back to the ALB — so the ALB
#   DNS name is internal and we don't publish it.
#
# Cut-over: while the EC2 module (deploy/terraform/) is still live, its
# Route 53 records own the apex. To cut over we MANUALLY swap the A
# record target (or edit zone_id-tracked DNS in the legacy module) and
# let TTLs drain before deleting the old records.
# ============================================================================

resource "aws_route53_record" "apex" {
  zone_id = data.aws_route53_zone.root.zone_id
  name    = var.domain
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.spa.domain_name
    zone_id                = aws_cloudfront_distribution.spa.hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "www" {
  zone_id = data.aws_route53_zone.root.zone_id
  name    = "www.${var.domain}"
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.spa.domain_name
    zone_id                = aws_cloudfront_distribution.spa.hosted_zone_id
    evaluate_target_health = false
  }
}

# AAAA records so IPv6 clients don't fall back to IPv4.
resource "aws_route53_record" "apex_ipv6" {
  zone_id = data.aws_route53_zone.root.zone_id
  name    = var.domain
  type    = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.spa.domain_name
    zone_id                = aws_cloudfront_distribution.spa.hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "www_ipv6" {
  zone_id = data.aws_route53_zone.root.zone_id
  name    = "www.${var.domain}"
  type    = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.spa.domain_name
    zone_id                = aws_cloudfront_distribution.spa.hosted_zone_id
    evaluate_target_health = false
  }
}
