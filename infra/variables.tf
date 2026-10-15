variable "region" {
  description = "AWS region. This account's organization SCP denies every region except eu-north-1."
  type        = string
  default     = "eu-north-1"
}

variable "ssh_public_key" {
  description = "Public key installed on all servers (tf.sh passes ~/.ssh/aims-rag.pub)."
  type        = string
}

variable "admin_cidr" {
  description = "CIDR allowed to SSH in. Empty = auto-detect your current public IP."
  type        = string
  default     = ""
}

variable "public_web_cidr" {
  description = "Who may reach the load balancer on port 80. Narrow it to your IP for a private demo."
  type        = string
  default     = "0.0.0.0/0"
}

# All three are free-plan eligible (t3.micro, t3.small, t4g.*, c7i-flex.large, m7i-flex.large).
variable "web_instance_type" {
  type    = string
  default = "t3.micro"
}

variable "api_instance_type" {
  type    = string
  default = "t3.small"
}

variable "db_instance_type" {
  type    = string
  default = "t3.micro"
}

variable "disk_gb" {
  type    = number
  default = 20
}
