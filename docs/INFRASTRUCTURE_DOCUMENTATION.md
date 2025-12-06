# Decay Memory App - DigitalOcean Infrastructure Documentation

## Overview

This document describes the current production deployment of the Decay Memory application on DigitalOcean.

---

## 1. DigitalOcean Droplet Specifications

| Property | Value |
|----------|-------|
| **Droplet Name** | srbAI01 |
| **Project** | first-project |
| **Region** | SFO3 (San Francisco 3) |
| **Image** | Ubuntu 24.04.3 LTS (Noble Numbat) |
| **Size** | 4 GB Memory / 2 vCPUs / 60 GB Disk |
| **Kernel** | 6.8.0-88-generic x86_64 |
| **Public IPv4** | 143.110.149.255 |
| **Private IPv4** | 10.124.0.2 |
| **VPC Network** | 10.48.0.5/16 (eth0) |
| **Estimated Cost** | ~$1.52/day (~$24/month) |

---

## 2. Network Configuration

### IP Addresses
```
Public IP:    143.110.149.255 (eth0)
Private IP:   10.124.0.2/20 (eth1)
VPC:          10.48.0.5/16 (eth0)
Docker:       172.17.0.1/16 (docker0)
Docker Net:   172.18.0.1/16 (br-5019a9377532)
```

### Listening Ports
| Port | Service | Description |
|------|---------|-------------|
| 22 | SSH | Secure Shell access |
| 53 | systemd-resolve | DNS resolution |
| 80 | docker-proxy | HTTP (redirects to HTTPS) |
| 443 | docker-proxy | HTTPS (frontend) |
| 6333 | Qdrant | Vector database (internal) |
| 8080 | Backend | Python API (internal) |

### Firewall Status
- **UFW**: Inactive (using Docker's iptables rules)
- **Docker** manages port forwarding via docker-proxy

---

## 3. Docker Architecture

### Docker Version
```
Docker version 28.2.2, build 28.2.2-0ubuntu1~24.04.1
```

### Running Containers

| Container Name | Image | Status | Ports | Purpose |
|---------------|-------|--------|-------|---------|
| decay_memory_ui | decay_memory_20-frontend | Up (unhealthy) | 0.0.0.0:80->80, 0.0.0.0:443->443 | React Frontend + Nginx |
| decay_memory_core | decay_memory_20-decay_memory_core | Up (healthy) | 8080/tcp (internal) | Python FastAPI Backend |
| decay_memory_db | qdrant/qdrant:latest | Up | 6333-6334/tcp (internal) | Qdrant Vector Database |

### Docker Networks
| Network Name | Driver | Scope |
|-------------|--------|-------|
| bridge | bridge | local |
| decay_memory_20_default | bridge | local |
| host | host | local |
| none | null | local |

### Docker Volumes
- `qdrant_storage` - Persistent storage for Qdrant vector database

---

## 4. Service Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              DigitalOcean Droplet                    │
                    │                  srbAI01                             │
                    │            143.110.149.255                           │
                    └─────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
              Port 80 (HTTP)                           Port 443 (HTTPS)
                    │                                         │
                    └──────────────┬──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     decay_memory_ui         │
                    │   (Nginx + React Frontend)  │
                    │     Container: frontend     │
                    │   Domain: permema.com       │
                    └──────────────┬──────────────┘
                                   │
                          /api/* requests
                                   │
                    ┌──────────────▼──────────────┐
                    │   decay_memory_core         │
                    │   (Python FastAPI Backend)  │
                    │      Internal Port: 8080    │
                    │   Uvicorn server:app        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     decay_memory_db         │
                    │    (Qdrant Vector DB)       │
                    │   Internal Ports: 6333-6334 │
                    │   Storage: qdrant_storage   │
                    └─────────────────────────────┘
```

---

## 5. SSL/TLS Configuration

### Certificate Details
| Property | Value |
|----------|-------|
| **Certificate Name** | permema.com |
| **Serial Number** | 658fa1b62563f3fcfd9a060a8c816006c9f |
| **Key Type** | ECDSA |
| **Domains** | permema.com |
| **Expiry Date** | 2026-03-04 21:40:07+00:00 (VALID: 89 days) |
| **Certificate Path** | /etc/letsencrypt/live/permema.com/fullchain.pem |
| **Private Key Path** | /etc/letsencrypt/live/permema.com/privkey.pem |

### SSL Settings (nginx.conf)
```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers HIGH:!aNULL:!MD5;
```

---

## 6. Docker Compose Configuration

The application is deployed from `decay_memory_app/Decay_Memory_2.0/` using docker-compose:

```yaml
version: '3.8'

services:
  # 1. DATABASE
  qdrant:
    image: qdrant/qdrant:latest
    container_name: decay_memory_db
    volumes:
      - qdrant_storage:/qdrant/storage
    expose:
      - "6333"
    restart: unless-stopped

  # 2. BACKEND
  decay_memory_core:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: decay_memory_core
    expose:
      - "8080"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENWEATHERMAP_API_KEY=${OPENWEATHERMAP_API_KEY}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    volumes:
      - ./flight_recorders:/app/flight_recorders
      - ./soul_state.json:/app/soul_state.json
      - ./heartbeat_mood.json:/app/heartbeat_mood.json
    depends_on:
      - qdrant
    restart: unless-stopped

  # 3. FRONTEND
  frontend:
    build: ./frontend
    container_name: decay_memory_ui
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - decay_memory_core
    restart: always

volumes:
  qdrant_storage:
```

---

## 7. Nginx Configuration (Frontend Container)

The frontend container runs Nginx with the following configuration:

```nginx
# 1. HTTP Redirect (Port 80 -> 443)
server {
    listen 80;
    server_name permema.com www.permema.com;
    
    # Redirect all traffic to HTTPS
    return 301 https://$host$request_uri;
}

# 2. HTTPS Server (Port 443)
server {
    listen 443 ssl;
    server_name permema.com www.permema.com;

    # SSL Keys (Mapped from Host)
    ssl_certificate /etc/letsencrypt/live/permema.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/permema.com/privkey.pem;

    # Modern SSL Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend Files
    root /usr/share/nginx/html;
    index index.html;

    # API Proxy (Forward /api/ requests to Python Backend)
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://decay_memory_core:8080;

        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health Check
    location /nginx-health {
        return 200 'OK';
        add_header Content-Type text/plain;
    }

    # SPA Fallback (For React Routing)
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

## 8. Domain Configuration

| Property | Value |
|----------|-------|
| **Domain** | permema.com |
| **www subdomain** | www.permema.com (redirects to permema.com) |
| **DNS Provider** | (configure A record pointing to 143.110.149.255) |
| **SSL Provider** | Let's Encrypt (Certbot) |

---

## 9. Persistent Data Locations

| Data Type | Location | Persistence |
|-----------|----------|-------------|
| Vector Database | Docker volume: `qdrant_storage` | Persisted |
| Flight Recorders | `./flight_recorders:/app/flight_recorders` | Bind mount |
| Soul State | `./soul_state.json:/app/soul_state.json` | Bind mount |
| Heartbeat Mood | `./heartbeat_mood.json:/app/heartbeat_mood.json` | Bind mount |
| SSL Certificates | `/etc/letsencrypt` (host) | Host filesystem |

---

## 10. System Services

| Service | Status | Description |
|---------|--------|-------------|
| docker.service | active (running) | Docker Application Container Engine |
| ssh.service | active (running) | OpenSSH Server |
| systemd-resolved | active (running) | DNS Resolution |

---

## 11. Management Commands

### Start/Stop Services
```bash
cd ~/decay_memory_app/Decay_Memory_2.0

# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart decay_memory_core
```

### View Container Status
```bash
docker ps -a
docker stats
```

### Access Container Shell
```bash
docker exec -it decay_memory_core /bin/bash
docker exec -it decay_memory_ui /bin/sh
docker exec -it decay_memory_db /bin/bash
```

### Renew SSL Certificate
```bash
certbot renew
# Restart frontend to pick up new certs
docker-compose restart frontend
```

### View Logs
```bash
# All containers
docker-compose logs -f

# Specific container
docker logs -f decay_memory_core
docker logs -f decay_memory_ui
docker logs -f decay_memory_db
```

---

## 12. Backup Recommendations

### Critical Data to Backup
1. **Qdrant Vector Database**
   ```bash
   docker run --rm -v qdrant_storage:/data -v $(pwd):/backup alpine tar cvf /backup/qdrant_backup.tar /data
   ```

2. **Application State Files**
   ```bash
   cp soul_state.json soul_state.json.backup
   cp heartbeat_mood.json heartbeat_mood.json.backup
   tar -cvf flight_recorders_backup.tar flight_recorders/
   ```

3. **Environment Variables**
   ```bash
   cp .env .env.backup
   ```

4. **SSL Certificates** (auto-renewed by Certbot)
   ```bash
   sudo tar -cvf letsencrypt_backup.tar /etc/letsencrypt/
   ```

---

## 13. Monitoring & Health Checks

### Health Endpoints
- **Frontend Health**: `https://permema.com/nginx-health`
- **Backend Health**: `https://permema.com/api/health`

### Docker Health Check Status
```bash
docker inspect --format='{{.State.Health.Status}}' decay_memory_core
docker inspect --format='{{.State.Health.Status}}' decay_memory_ui
```

### Resource Monitoring
```bash
# CPU, Memory, Disk
htop
df -h
free -m

# Docker stats
docker stats --no-stream
```

---

## 14. Troubleshooting

### Common Issues

1. **Frontend shows "unhealthy"**
   - Check nginx logs: `docker logs decay_memory_ui`
   - Verify SSL certs are mounted correctly
   - Check if backend is reachable

2. **API requests failing**
   - Check backend logs: `docker logs decay_memory_core`
   - Verify Qdrant is running: `docker logs decay_memory_db`
   - Check network connectivity between containers

3. **SSL Certificate Issues**
   - Renew cert: `certbot renew`
   - Check cert expiry: `certbot certificates`
   - Restart frontend after renewal

4. **Out of Disk Space**
   - Clean unused Docker images: `docker image prune -a`
   - Check disk usage: `df -h`
   - Remove old logs: `docker system prune`

---

*Document generated: December 2024*
*Last infrastructure review: December 5, 2025*
