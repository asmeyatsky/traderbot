# ============================================================================
# Outputs
# ============================================================================

output "elastic_ip" {
  description = "Public Elastic IP of the EC2 instance"
  value       = aws_eip.app.public_ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh ubuntu@${aws_eip.app.public_ip}"
}

output "nameservers" {
  description = "Route 53 nameservers — update your domain registrar with these"
  value       = aws_route53_zone.main.name_servers
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.app.id
}
