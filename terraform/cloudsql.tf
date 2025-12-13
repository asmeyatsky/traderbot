# Cloud SQL Instance
resource "google_sql_database_instance" "postgres" {
  name             = "traderbot-db-instance-unique-${random_id.db_name_suffix.hex}"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier = "db-f1-micro"
    
    ip_configuration {
      ipv4_enabled    = true
      private_network = google_compute_network.vpc.id
    }
  }

  deletion_protection = false
  depends_on          = [google_service_networking_connection.private_vpc_connection]
}

resource "random_id" "db_name_suffix" {
  byte_length = 4
  keepers = {
    project_id = var.project_id
    region     = var.region
  }
}

# Database
resource "google_sql_database" "database" {
  name     = "traderbot"
  instance = google_sql_database_instance.postgres.name
}

# Database User
resource "google_sql_user" "users" {
  name     = "trading"
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
}

variable "db_password" {
  description = "The password for the database user"
  type        = string
  sensitive   = true
  default     = "trading-password-change-me"
}
