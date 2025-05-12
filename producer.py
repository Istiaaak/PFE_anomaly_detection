import time, argparse
from pathlib import Path
from kafka import KafkaProducer
from data.data import MVTecDataset

TOPIC            = "patchcore_images"
BOOTSTRAP_SERVERS = ["localhost:9092"]

def create_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: v
    )

def collect_mvtc_images(dataset_root: Path):
    imgs = []
    test_root = dataset_root / "test"
    imgs += list((test_root/"good").glob("*.png"))
    for sub in test_root.iterdir():
        if sub.is_dir() and sub.name!="good":
            imgs += list(sub.glob("*.png"))
    return imgs

def collect_folder_images(folder: Path):
    return list(folder.glob("*.png")) + list(folder.glob("*.jpg"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use-mvtc", action="store_true")
    p.add_argument("--folder", required=True)
    p.add_argument("--delay", type=float, default=1.0)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--once",   action="store_true")
    grp.add_argument("--epochs", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    folder = Path(args.folder)
    imgs = collect_mvtc_images(folder) if args.use_mvtc else collect_folder_images(folder)
    if not imgs:
        raise RuntimeError(f"Aucune image dans {folder}")
    producer = create_producer()
    print(f"Start: {len(imgs)} images × { (1 if args.once else args.epochs) } passes", flush=True)

    def pass_cycle():
        for i,path in enumerate(imgs):
            producer.send(TOPIC, key=path.name.encode(), value=path.read_bytes())
            print(f"[{i}] Produced {path.name}", flush=True)
            time.sleep(args.delay)

    if args.once:
        pass_cycle()
    else:
        for e in range(args.epochs):
            print(f"Epoch {e+1}/{args.epochs}", flush=True)
            pass_cycle()
    print("Producer terminé", flush=True)

if __name__=="__main__":
    main()
