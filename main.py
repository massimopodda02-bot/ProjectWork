# --- SCRIPT UNIFICATO PROJECT WORK L-31 ---

# 1. PULIZIA TOTALE DI VECCHI PROCESSI
!pkill -f streamlit

# 2. INSTALLAZIONE DELLE LIBRERIE
print("Installazione librerie in corso...")
!pip install -q streamlit pandas numpy scikit-learn seaborn matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import subprocess
import time
from google.colab.output import eval_js

# 3. GENERAZIONE DEL DATASET (450 record)
np.random.seed(42)
n_record = 450
zone = ["Nord", "Sud", "Centro", "Isole"]
categorie = ["Consegna Rapida", "Assistenza Tech", "Manutenzione", "Consulenza"]
data_oggi = datetime(2026, 11, 1)

id_clienti, lista_zone, date_iscrizione, num_richieste, importo_speso, giorni_ultima_richiesta, cat_preferita = [], [], [], [], [], [], []
segmenti = ["Molto Attivo", "Occasionale", "Recente", "Inattivo"]
pesi_segmenti = [0.20, 0.35, 0.20, 0.25]

for i in range(1, n_record + 1):
    id_clienti.append(f"C{i:03d}")
    lista_zone.append(np.random.choice(zone))
    cat_preferita.append(np.random.choice(categorie))
    segmento = np.random.choice(segmenti, p=pesi_segmenti)
    
    if segmento == "Molto Attivo":
        n_req, spesa, giorni_ult, giorni_iscr = np.random.randint(15, 30), np.random.uniform(800.0, 2000.0), np.random.randint(1, 15), np.random.randint(180, 365)
    elif segmento == "Occasionale":
        n_req, spesa, giorni_ult, giorni_iscr = np.random.randint(5, 12), np.random.uniform(300.0, 900.0), np.random.randint(10, 45), np.random.randint(120, 365)
    elif segmento == "Recente":
        n_req, spesa, giorni_ult, giorni_iscr = np.random.randint(1, 6), np.random.uniform(50.0, 350.0), np.random.randint(1, 10), np.random.randint(1, 45)
    else:
        n_req, spesa, giorni_ult, giorni_iscr = np.random.randint(1, 4), np.random.uniform(20.0, 200.0), np.random.randint(60, 180), np.random.randint(200, 365)

    date_iscrizione.append((data_oggi - timedelta(days=int(giorni_iscr))).strftime("%Y-%m-%d"))
    num_richieste.append(n_req)
    importo_speso.append(round(spesa, 2))
    giorni_ultima_richiesta.append(giorni_ult)

df = pd.DataFrame({
    "identificativo": id_clienti, "zona": lista_zone, "data_iscrizione": date_iscrizione,
    "numero_richieste": num_richieste, "importo_totale_speso": importo_speso,
    "giorni_ultima_richiesta": giorni_ultima_richiesta, "categoria_servizio_preferita": cat_preferita
})
df.to_csv("clienti_startup.csv", index=False, encoding="utf-8")

# 4. PREPARAZIONE DATI E VALUTAZIONE SILHOUETTE (K=3, K=4, K=5)
feature_cols = ["numero_richieste", "importo_totale_speso", "giorni_ultima_richiesta"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

print("\n--- VALUTAZIONE INDICE DI SILHOUETTE ---")
for k_val in [3, 4, 5]:
    km_test = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    labels_test = km_test.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_test)
    print(f"-> K = {k_val} | Indice di Silhouette: {score:.4f}")

# Applicazione finale con K = 4
kmeans_finale = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans_finale.fit_predict(X_scaled)
df.to_csv("clienti_startup_con_cluster.csv", index=False, encoding="utf-8")
print("Dataset con cluster salvato con successo!\n")

# 5. SCRITTURA DEL FILE app.py PER LA DASHBOARD STREAMLIT
app_code = '''import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dashboard Segmentazione Clienti", layout="wide")
st.title("📊 Dashboard Interattiva - Segmentazione Clienti Startup")
st.markdown("Prototipo di analisi comportamentale e clustering della clientela (Corso di Laurea L-31).")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("clienti_startup_con_cluster.csv")
    except FileNotFoundError:
        df = pd.read_csv("clienti_startup.csv")
        scaler = StandardScaler()
        X = df[["numero_richieste", "importo_totale_speso", "giorni_ultima_richiesta"]]
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
    return df

df = load_data()

st.sidebar.header("🔍 Filtri Interattivi")
zone_disponibili = ["Tutte"] + list(df['zona'].unique())
zona_selezionata = st.sidebar.selectbox("Seleziona Zona", zone_disponibili)

cat_disponibili = ["Tutte"] + list(df['categoria_servizio_preferita'].unique())
cat_selezionata = st.sidebar.selectbox("Categoria di Servizio Preferita", cat_disponibili)

min_giorni = int(df['giorni_ultima_richiesta'].min())
max_giorni = int(df['giorni_ultima_richiesta'].max())
periodo_selezionato = st.sidebar.slider("Filtra per Periodo (Giorni max dall'ultima richiesta)", min_value=min_giorni, max_value=max_giorni, value=(min_giorni, max_giorni))

df_filtrato = df.copy()
if zona_selezionata != "Tutte":
    df_filtrato = df_filtrato[df_filtrato['zona'] == zona_selezionata]
if cat_selezionata != "Tutte":
    df_filtrato = df_filtrato[df_filtrato['categoria_servizio_preferita'] == cat_selezionata]

df_filtrato = df_filtrato[(df_filtrato['giorni_ultima_richiesta'] >= periodo_selezionato[0]) & (df_filtrato['giorni_ultima_richiesta'] <= periodo_selezionato[1])]
st.sidebar.markdown(f"**Clienti visualizzati:** {len(df_filtrato)} su {len(df)}")

st.subheader("👥 Profilo dei Gruppi Individuati")
feature_cols = ["numero_richieste", "importo_totale_speso", "giorni_ultima_richiesta"]
profilo_gruppi = df_filtrato.groupby('cluster')[feature_cols].agg(['mean', 'count']).round(2)
st.dataframe(profilo_gruppi, use_container_width=True)

st.subheader("📈 Analisi Visuale dei Cluster")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_filtrato, x="numero_richieste", y="importo_totale_speso", hue="cluster", palette="viridis", s=80, ax=ax, alpha=0.8)
    ax.set_title("Richieste vs Importo Speso")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
with col2:
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_filtrato, x="giorni_ultima_richiesta", y="importo_totale_speso", hue="cluster", palette="viridis", s=80, ax=ax2, alpha=0.8)
    ax2.set_title("Recenza vs Importo Speso")
    ax2.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig2)

st.markdown("---")
st.subheader("🔮 Classificazione Automatica di un Nuovo Cliente")
with st.form("form_nuovo_cliente"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        nuove_richieste = st.number_input("Numero Richieste", min_value=1, max_value=50, value=12)
    with col_b:
        nuova_spesa = st.number_input("Importo Totale Speso (€)", min_value=10.0, max_value=5000.0, value=750.0)
    with col_c:
        nuovi_giorni = st.number_input("Giorni dall'Ultima Richiesta", min_value=0, max_value=365, value=10)
    submit_btn = st.form_submit_button("Ottieni Gruppo di Appartenenza")

if submit_btn:
    scaler = StandardScaler()
    X_all = df[["numero_richieste", "importo_totale_speso", "giorni_ultima_richiesta"]]
    X_scaled = scaler.fit_transform(X_all)
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)
    nuovo_record = np.array([[nuove_richieste, nuova_spesa, nuevos_giorni if 'nuevos_giorni' in locals() else nuovi_giorni]])
    nuovo_scaled = scaler.transform(nuovo_record)
    cluster_predetto = kmeans_model.predict(nuovo_scaled)[0]
    st.success(f"✅ Il nuovo cliente è associato al **Cluster {cluster_predetto}**!")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

# 6. AVVIO STREAMLIT
print("Avvio della dashboard Streamlit...")
subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"])
time.sleep(4)

proxy_url = eval_js("google.colab.kernel.proxyPort(8501)")
print(f"\n✨ Fatto! Clicca su questo link per aprire la dashboard interattiva:\n{proxy_url}")