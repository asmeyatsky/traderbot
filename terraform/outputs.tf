output "vpc_connector_name" {
  value = google_vpc_access_connector.connector.name
}

output "db_instance_connection_name" {
  value = google_sql_database_instance.postgres.connection_name
}

output "db_user" {
  value = google_sql_user.users.name
}

output "db_password" {
  value     = google_sql_user.users.password
  sensitive = true
}

output "redis_host" {
  value = google_redis_instance.cache.host
}

output "redis_port" {
  value = google_redis_instance.cache.port
}

output "artifact_registry_repo" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.name}"
}
