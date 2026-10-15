output "app_url" {
  value = "http://${aws_lb.main.dns_name}"
}

output "admin_cidr" {
  value = local.admin_cidr
}

output "servers" {
  value = {
    for role, s in aws_instance.server : role => {
      public_ip  = s.public_ip
      private_ip = s.private_ip
    }
  }
}
