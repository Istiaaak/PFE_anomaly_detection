from kafka import KafkaConsumer, KafkaProducer

SERVERS = ["localhost:9092"]

def create_producer(topic: str):
    return KafkaProducer(
        bootstrap_servers=SERVERS,
        value_serializer=lambda v: v
    )

def create_consumer(
    topic:    str,
    group_id: str = "patchcore_consumer",
    servers:  list = None
):

    if servers is None:
        servers = SERVERS

    return KafkaConsumer(
        topic,
        bootstrap_servers=servers,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id=group_id,
        value_deserializer=lambda v: v
    )
