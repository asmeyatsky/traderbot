# Memorystore (Redis)
resource "google_redis_instance" "cache" {
  name               = "traderbot-cache"
  memory_size_gb     = 1
  tier               = "BASIC"
  
  # Ensure we pick a valid zone in the region (e.g. europe-west2-a)
  # Keeping it simple by letting GCloud pick or appending 'a'? 
  # Actually, location_id handles zone. If not specified, standard tier picks automatically, basic might too.
  # But let's be safe and use region + "-a"
  location_id        = "${var.region}-a"
  
  authorized_network = google_compute_network.vpc.id
  connect_mode       = "DIRECT_PEERING"

  redis_version      = "REDIS_7_0"
  display_name       = "Traderbot Redis Cache"

  depends_on = [google_service_networking_connection.private_vpc_connection]
}
