# KeyDB (Redis-compatible open source alternative)
# Deployed as GCE instance with KeyDB Docker container
resource "google_compute_instance" "keydb" {
  name         = "traderbot-keydb"
  machine_type = "e2-micro"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 10
    }
  }

  network_interface {
    network = google_compute_network.vpc.id
    access_config {
      # Ephemeral IP for management
    }
  }

  metadata = {
    startup-script = <<-EOT
      #!/bin/bash
      apt-get update
      apt-get install -y docker.io
      
      # Run KeyDB container
      docker run -d \
        --name keydb \
        -p 6379:6379 \
        --restart unless-stopped \
        eqalpha/keydb:latest \
        keydb-server --appendonly yes --protected-mode no
    EOT
  }

  tags = ["keydb"]

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Firewall rule for KeyDB
resource "google_compute_firewall" "keydb" {
  name    = "allow-keydb"
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
    ports    = ["6379"]
  }

  source_tags = ["trading-api"]
  target_tags = ["keydb"]
}
