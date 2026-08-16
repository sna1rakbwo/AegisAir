# Runtime image for the AegisAir PX4 adapter.
#
# The base image already contains ROS 2 Humble, px4_msgs, and the
# Micro-XRCE-DDS Agent.  This layer only adds the Python MQTT client used by
# ``px4_adapter/node.py``; the adapter source itself is mounted at runtime so
# code stays in the repository.
FROM safety-margin-gate0-ros2:humble

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && pip3 install --no-cache-dir paho-mqtt \
    && rm -rf /var/lib/apt/lists/*

COPY px4_adapter /work/px4_adapter
COPY px4_adapter/docker/run_adapter.sh /usr/local/bin/run_adapter.sh
RUN chmod +x /usr/local/bin/run_adapter.sh

ENTRYPOINT ["/usr/local/bin/run_adapter.sh"]
