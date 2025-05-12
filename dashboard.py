import os
import time
import io
import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from data.data import MVTecDataset, mvtec_classes, DEFAULT_SIZE
from utils.utils import backbones
from utils.utils_app import load_patchcore_model, tensor_to_img
from logger import get_logger

os.makedirs("logs", exist_ok=True)
logger = get_logger("streamlit", logfile="logs/patchcore.log")


def main():
    st.set_page_config(page_title="PatchCore Live Dashboard", layout="wide")

    cls           = st.sidebar.selectbox("Classe MVTec", mvtec_classes(), key="cls")
    backbone_key  = st.sidebar.selectbox("Backbone", list(backbones.keys()), key="backbone")
    use_cache     = st.sidebar.checkbox("Charger memory bank existante", True, key="use_cache")
    f_coreset     = st.sidebar.slider("Fraction coreset", 0.0, 1.0, 0.1, 0.01, key="f_coreset")
    eps           = st.sidebar.slider("Epsilon coreset", 0.01, 1.0, 0.9, 0.01, key="eps")
    k_nn          = st.sidebar.number_input("k-nearest", 1, 10, 3, key="k_nn")
    use_kafka     = st.sidebar.checkbox("Utiliser Kafka (local)", False, key="use_kafka")

    vanilla = (backbone_key == "WideResNet50")
    size_map = {
        'WideResNet50': DEFAULT_SIZE,
        'ResNet50':     224,
        'ResNet50-4':   288,
        'ResNet50-16':  384,
        'ResNet101':    224
    }
    image_size = size_map[backbone_key]

    if "bank_built" not in st.session_state:
        st.session_state.bank_built = False

    if not st.session_state.bank_built:
        st.sidebar.markdown("## Étape 1 : Construire la memory bank")
        if st.sidebar.button("Construire la memory bank"):

            placeholder = st.sidebar.empty()
            def cb(i, total):
                placeholder.progress(i / total)

            model, train_scores = load_patchcore_model(
                cls, backbone_key, f_coreset, eps, k_nn, use_cache,
                progress_callback=cb
            )
            placeholder.empty()

            st.session_state.model        = model
            st.session_state.train_scores = train_scores
            st.session_state.bank_built   = True
            st.sidebar.success("Memory bank prête !")

            st.rerun()

        st.stop()


    default_thresh = float(np.percentile(st.session_state.train_scores, 99))
    seuil = st.sidebar.slider(
        "Seuil d’anomalie",
        float(st.session_state.train_scores.min()),
        float(st.session_state.train_scores.max()),
        default_thresh,
        key="seuil"
    )

    if "running" not in st.session_state:
        st.session_state.running = False
        st.session_state.idx     = 0
        st.session_state.scores  = []

    if st.sidebar.button("Démarrer le test"):
        st.session_state.running    = True
        st.session_state.start_time = time.time()
    if st.sidebar.button("Arrêter le test"):
        st.session_state.running    = False


    ph_kpi    = [c.empty() for c in st.columns(4)]
    col_img, col_map = st.columns(2)
    ph_img    = col_img.empty()
    ph_map    = col_map.empty()
    ch1, ch2  = st.columns(2)
    ph_chart1 = ch1.empty()
    ph_chart2 = ch2.empty()
    ph_pie    = st.empty()


    if st.session_state.running:
        model   = st.session_state.model

        if use_kafka:
            from utils.utils_kafka import create_consumer
            consumer = create_consumer("patchcore_images", group_id="streamlit_consumer")
            stream = consumer
        else:
            _, test_ds = MVTecDataset(cls, size=image_size, vanilla=vanilla).get_datasets()
            stream = range(len(test_ds))

        for item in stream:
            if not st.session_state.running:
                break

            if use_kafka:
                msg = item
                img = Image.open(io.BytesIO(msg.value)).convert("RGB")
                img = img.resize((image_size, image_size))
                transform = MVTecDataset(cls, size=image_size, vanilla=vanilla)\
                             .get_datasets()[0].transform
                sample = transform(img).unsqueeze(0)
                img_np = np.array(img) / 255.0
            else:
                sample, _, _ = test_ds[item]
                img_np       = tensor_to_img(sample.squeeze(0), vanilla)
                sample       = sample.unsqueeze(0)

            score, amap = model.predict(sample)
            val         = float(score.item())
            st.session_state.scores.append(val)
            st.session_state.idx += 1

            mean    = np.mean(st.session_state.scores)
            elapsed = time.time() - st.session_state.start_time
            total   = (len(test_ds) if not use_kafka else st.session_state.idx)

            ph_kpi[0].metric("Images traitées", f"{st.session_state.idx}/{total}")
            ph_kpi[1].metric("Temps écoulé",    f"{elapsed:.2f}s")
            ph_kpi[2].metric("Score moyen",     f"{mean:.3f}")
            ph_kpi[3].metric("Seuil actif",     f"{seuil:.3f}")

            ph_img.image(img_np, caption="Image originale", use_container_width=True)

            amap_np   = amap.squeeze().cpu().numpy()
            amap_norm = (amap_np - amap_np.min())/(amap_np.max()-amap_np.min()+1e-8)
            fig = go.Figure([
                go.Image(z=(img_np*255).astype("uint8")),
                go.Heatmap(z=amap_norm, colorscale="Jet", opacity=0.5)
            ])
            fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=350)
            ph_map.plotly_chart(fig, use_container_width=True)

            fig1 = go.Figure([go.Scatter(
                x=list(range(len(st.session_state.scores))),
                y=st.session_state.scores,
                mode="lines+markers"
            )])
            fig1.add_hline(y=seuil, line_color="red")
            ph_chart1.plotly_chart(fig1, use_container_width=True)

            fig2 = px.histogram(st.session_state.scores, nbins=20)
            fig2.add_vline(x=seuil, line_dash="dash", line_color="red")
            ph_chart2.plotly_chart(fig2, use_container_width=True)

            normal = sum(s < seuil for s in st.session_state.scores)
            anom   = sum(s >= seuil for s in st.session_state.scores)
            fig3   = px.pie(values=[normal, anom], names=["Normal","Anomalie"])
            ph_pie.plotly_chart(fig3, use_container_width=True)

            time.sleep(5)

        st.session_state.running = False


if __name__ == "__main__":
    main()