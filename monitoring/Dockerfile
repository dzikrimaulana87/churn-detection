FROM prom/prometheus

# Salin file konfigurasi Prometheus ke image
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Salin file konfigurasi tambahan jika ada
COPY prometheus.config /etc/prometheus/prometheus.config

# Default port Prometheus adalah 9090
EXPOSE 9090

CMD [ "--config.file=/etc/prometheus/prometheus.yml" ]
