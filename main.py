# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
import time
import os

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
print("✅ Library berhasil dimuat.")

# =============================================================================
# SETUP OUTPUT DIRECTORY
# =============================================================================
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ Folder output dibuat: {output_dir}/")

# =============================================================================
# BAGIAN 1: PERSIAPAN DATA
# =============================================================================
print("\n--- BAGIAN 1: PERSIAPAN DATA ---")

nama_file = "dataset/dataset_update.csv" 
kolom_fitur = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
X_scaled = None
df_clean = None
scaler = MinMaxScaler()
X_pca = None

try:
    df = pd.read_csv(nama_file)
    df_fitur = df[kolom_fitur].copy()
    
    for col in kolom_fitur:
        df_fitur[col] = pd.to_numeric(df_fitur[col], errors='coerce')
    
    df_fitur.fillna(df_fitur.median(), inplace=True)
    df_clean = df_fitur.drop_duplicates().copy()
    X_scaled = scaler.fit_transform(df_clean)
    
    print(f"✅ Data bersih siap: {X_scaled.shape[0]} baris, {X_scaled.shape[1]} fitur")

except FileNotFoundError:
    print(f"❌ ERROR: File '{nama_file}' tidak ditemukan. Harap periksa path file Anda.")
    X_scaled = None
except Exception as e:
    print(f"❌ ERROR: Gagal memuat data: {e}")
    X_scaled = None

# =============================================================================
# BAGIAN 2: AUTO-TUNING HEAD-TO-HEAD (K-Means vs Mini-Batch)
# =============================================================================
if X_scaled is not None:
    print("\n--- BAGIAN 2: MEMBANDINGKAN AUTO-TUNING (K-Means vs Mini-Batch) ---")
    
    wcss_kmeans = []
    wcss_minibatch = []
    sil_kmeans = []
    sil_minibatch = []
    time_kmeans = []
    time_minibatch = []
    
    range_k = range(2, 11)
    
    use_sampling = len(X_scaled) > 10000
    X_sample = None
    if use_sampling:
        print("Mempersiapkan data sampel (10k) untuk evaluasi Silhouette...")
        np.random.seed(42)
        sample_indices = np.random.choice(X_scaled.shape[0], 10000, replace=False)
        X_sample = X_scaled[sample_indices]
    else:
        X_sample = X_scaled
            
    print("Memulai iterasi k=2 sampai 10 untuk kedua algoritma...")
    
    for k in range_k:
        start_t = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        end_t = time.time()
        
        wcss_kmeans.append(kmeans.inertia_)
        time_kmeans.append(end_t - start_t)
        if use_sampling:
            labels_sample_km = kmeans.predict(X_sample)
            sil_kmeans.append(silhouette_score(X_sample, labels_sample_km))
        else:
            sil_kmeans.append(silhouette_score(X_scaled, kmeans.labels_))
        
        start_t = time.time()
        mbkm = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto', batch_size=1024)
        mbkm.fit(X_scaled)
        end_t = time.time()
        
        wcss_minibatch.append(mbkm.inertia_)
        time_minibatch.append(end_t - start_t)
        if use_sampling:
            labels_sample_mbkm = mbkm.predict(X_sample)
            sil_minibatch.append(silhouette_score(X_sample, labels_sample_mbkm))
        else:
            sil_minibatch.append(silhouette_score(X_scaled, mbkm.labels_))
            
        print(f"  k={k} selesai. (KMeans: {time_kmeans[-1]:.2f}s, MiniBatch: {time_minibatch[-1]:.2f}s)")
            
    print("✅ Perhitungan auto-tuning komparatif selesai.")

    # =============================================================================
    # BAGIAN 3: VISUALISASI PERBANDINGAN (ELBOW & SILHOUETTE)
    # =============================================================================
    print("\n--- BAGIAN 3: VISUALISASI HASIL PERBANDINGAN ---")

    plt.figure(figsize=(12, 7))
    plt.plot(range_k, wcss_kmeans, 'o-', markerfacecolor='blue', markersize=8, label='K-Means (Standar)')
    plt.plot(range_k, wcss_minibatch, 's--', markerfacecolor='green', markersize=8, label='Mini-Batch K-Means')
    plt.title('Perbandingan Grafik Elbow Method')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(range_k)
    plt.savefig(os.path.join(output_dir, "comparison_auto_elbow_method.png"))
    print(f"✅ Grafik perbandingan Elbow disimpan di folder output/")
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.plot(range_k, sil_kmeans, 'o-', markerfacecolor='blue', markersize=8, label='K-Means (Standar)')
    plt.plot(range_k, sil_minibatch, 's--', markerfacecolor='green', markersize=8, label='Mini-Batch K-Means')
    plt.title('Perbandingan Silhouette Score')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(range_k)
    plt.savefig(os.path.join(output_dir, "comparison_auto_silhouette_score.png"))
    print(f"✅ Grafik perbandingan Silhouette disimpan di folder output/")
    plt.close()

    # =============================================================================
    # BAGIAN 4: TABEL PERBANDINGAN FINAL
    # =============================================================================
    print("\n--- BAGIAN 4: TABEL PERBANDINGAN KINERJA ---")
    
    df_perbandingan = pd.DataFrame({
        'k': range_k,
        'WCSS_KMeans': wcss_kmeans,
        'WCSS_MiniBatch': wcss_minibatch,
        'Silhouette_KMeans': sil_kmeans,
        'Silhouette_MiniBatch': sil_minibatch,
        'Waktu_KMeans (detik)': time_kmeans,
        'Waktu_MiniBatch (detik)': time_minibatch
    })
    
    total_time_kmeans = sum(time_kmeans)
    total_time_minibatch = sum(time_minibatch)

    print("\n" + "+" + "-"*110 + "+")
    print("| TABEL PERBANDINGAN KINERJA AUTO-TUNING (k=2 s/d 10)                                                               |")
    print("+" + "-"*110 + "+")
    print(df_perbandingan.to_string(float_format="%.4f", index=False))
    print("+" + "-"*110 + "+")

    print("\nRingkasan Waktu Komputasi (Total untuk k=2 s/d 10):")
    print(f"  - K-Means Standar  : {total_time_kmeans:.4f} detik")
    print(f"  - Mini-Batch K-Means : {total_time_minibatch:.4f} detik")
    
    best_k_kmeans = int(df_perbandingan.loc[df_perbandingan['Silhouette_KMeans'].idxmax(), 'k'])
    best_k_minibatch = int(df_perbandingan.loc[df_perbandingan['Silhouette_MiniBatch'].idxmax(), 'k'])

    print(f"\n💡 Rekomendasi k (Silhouette Tertinggi) K-Means       : k = {best_k_kmeans}")
    print(f"💡 Rekomendasi k (Silhouette Tertinggi) Mini-Batch  : k = {best_k_minibatch}")
    
    df_perbandingan.to_csv(os.path.join(output_dir, "hasil_tabel_perbandingan_tuning.csv"), index=False)
    print("💾 Tabel perbandingan lengkap disimpan di folder output/")

    # =============================================================================
    # BAGIAN 5: VISUALISASI PCA FINAL
    # =============================================================================
    print("\n--- BAGIAN 5: VISUALISASI PCA FINAL ---")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print("✅ PCA selesai.")

    k_untuk_kmeans = best_k_kmeans
    print(f"Menjalankan K-Means final untuk PCA dengan k={k_untuk_kmeans}...")
    
    kmeans_final = KMeans(n_clusters=k_untuk_kmeans, random_state=42, n_init='auto')
    kmeans_final.fit(X_scaled)
    labels_kmeans_final = kmeans_final.labels_
    centroids_kmeans_final_scaled = kmeans_final.cluster_centers_

    palette_km = sns.color_palette("tab10", k_untuk_kmeans)
    plt.figure(figsize=(10, 7))
    df_plot_km = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_plot_km['cluster'] = labels_kmeans_final
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette=palette_km, data=df_plot_km, legend="full", alpha=0.5, s=50)
    
    centroids_pca_km = pca.transform(centroids_kmeans_final_scaled)
    plt.scatter(centroids_pca_km[:, 0], centroids_pca_km[:, 1], marker='X', s=200, c=palette_km, edgecolor='k', label='Centroids (K-Means)')
    
    plt.title(f'Sebaran Cluster K-Means (k={k_untuk_kmeans})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_filename_km = f"comparison_auto_pca_kmeans_k{k_untuk_kmeans}.png"
    plt.savefig(os.path.join(output_dir, output_filename_km))
    print(f"✅ Visualisasi PCA K-Means disimpan di folder output/")
    plt.close()

    k_untuk_mbkm = best_k_minibatch
    print(f"Menjalankan Mini-Batch final untuk PCA dengan k={k_untuk_mbkm}...")
    
    mbkm_final = MiniBatchKMeans(n_clusters=k_untuk_mbkm, random_state=42, n_init='auto', batch_size=1024)
    mbkm_final.fit(X_scaled)
    labels_mbkm_final = mbkm_final.labels_
    centroids_mbkm_final_scaled = mbkm_final.cluster_centers_

    palette_mbkm = sns.color_palette("tab10", k_untuk_mbkm)
    plt.figure(figsize=(10, 7))
    df_plot_mbkm = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_plot_mbkm['cluster'] = labels_mbkm_final
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette=palette_mbkm, data=df_plot_mbkm, legend="full", alpha=0.5, s=50)
    
    centroids_pca_mbkm = pca.transform(centroids_mbkm_final_scaled)
    plt.scatter(centroids_pca_mbkm[:, 0], centroids_pca_mbkm[:, 1], marker='s', s=200, c=palette_mbkm, edgecolor='k', label='Centroids (Mini-Batch)')
    
    plt.title(f'Sebaran Cluster Mini-Batch K-Means (k={k_untuk_mbkm})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_filename_mbkm = f"comparison_auto_pca_minibatch_k{k_untuk_mbkm}.png"
    plt.savefig(os.path.join(output_dir, output_filename_mbkm))
    print(f"✅ Visualisasi PCA Mini-Batch disimpan di folder output/")
    plt.close()

    # =============================================================================
    # BAGIAN 6: SIMPAN HASIL PROFIL & DATA
    # =============================================================================
    print("\n--- BAGIAN 6: MENYIMPAN HASIL PROFIL & DATA ---")
    
    df_profil_kmeans = pd.DataFrame(scaler.inverse_transform(centroids_kmeans_final_scaled), columns=kolom_fitur)
    df_profil_kmeans['Jumlah_Anggota'] = pd.Series(labels_kmeans_final).value_counts().sort_index()
    df_profil_kmeans.index.name = f'Cluster (k={k_untuk_kmeans})'
    print(f"\n📊 Profil K-Means (k={k_untuk_kmeans}):")
    print(df_profil_kmeans.to_string(float_format="%.2f"))

    df_profil_mbkm = pd.DataFrame(scaler.inverse_transform(centroids_mbkm_final_scaled), columns=kolom_fitur)
    df_profil_mbkm['Jumlah_Anggota'] = pd.Series(labels_mbkm_final).value_counts().sort_index()
    df_profil_mbkm.index.name = f'Cluster (k={k_untuk_mbkm})'
    print(f"\n📊 Profil Mini-Batch K-Means (k={k_untuk_mbkm}):")
    print(df_profil_mbkm.to_string(float_format="%.2f"))

    file_profil_final = "comparison_auto_profil_klaster.xlsx"
    with pd.ExcelWriter(os.path.join(output_dir, file_profil_final)) as writer:
        df_profil_kmeans.to_excel(writer, sheet_name=f'Profil_KMeans_k{k_untuk_kmeans}')
        df_profil_mbkm.to_excel(writer, sheet_name=f'Profil_MiniBatch_k{k_untuk_mbkm}')
    print(f"💾 Profil klaster disimpan di folder output/")
    
    df_clean[f'cluster_kmeans_k{k_untuk_kmeans}'] = labels_kmeans_final
    df_clean[f'cluster_minibatch_k{k_untuk_mbkm}'] = labels_mbkm_final
    
    file_data_final = "comparison_auto_data_labeled.csv"
    df_clean.to_csv(os.path.join(output_dir, file_data_final), index=False)
    print(f"💾 Data final dengan label disimpan di folder output/")

    # =============================================================================
    # BAGIAN 7: ANALISIS KONTEKSTUAL (PERBAIKAN)
    # =============================================================================
    print("\n--- BAGIAN 7: ANALISIS KONTEKSTUAL ---")

    # --- PERBAIKAN: Menggunakan .loc[df_clean.index] untuk mencocokkan baris ---
    # Kita harus mengambil data dari df asli HANYA pada baris yang tersisa di df_clean
    df_clean['parameter_pencemar_kritis'] = df.loc[df_clean.index, 'parameter_pencemar_kritis']
    df_clean['kategori'] = df.loc[df_clean.index, 'kategori']
    df_clean['stasiun'] = df.loc[df_clean.index, 'stasiun']
    df_clean['periode'] = pd.to_datetime(df.loc[df_clean.index, 'periode'], format='%d/%m/%Y') # Format disesuaikan
    df_clean['bulan'] = df_clean['periode'].dt.month

    # Analysis by station
    print("\n📍 Distribusi Cluster per Stasiun:")
    station_dist = pd.crosstab(df_clean[f'cluster_kmeans_k{k_untuk_kmeans}'], 
                            df_clean['stasiun'], 
                            normalize='columns') * 100
    print(station_dist.round(1))

    # Analysis by official category
    print("\n🏷️ Distribusi Cluster per Kategori:")
    kategori_dist = pd.crosstab(df_clean[f'cluster_kmeans_k{k_untuk_kmeans}'], 
                                df_clean['kategori'], 
                                normalize='columns') * 100
    print(kategori_dist.round(1))

    # Analysis by critical parameter
    print("\n⚠️ Distribusi Cluster per Parameter Pencemar Kritis:")
    param_dist = pd.crosstab(df_clean[f'cluster_kmeans_k{k_untuk_kmeans}'], 
                            df_clean['parameter_pencemar_kritis'], 
                            normalize='columns') * 100
    print(param_dist.round(1))

    # Save to Excel
    with pd.ExcelWriter(os.path.join(output_dir, "analisis_kontekstual.xlsx")) as writer:
        station_dist.to_excel(writer, sheet_name='Per_Stasiun')
        kategori_dist.to_excel(writer, sheet_name='Per_Kategori')
        param_dist.to_excel(writer, sheet_name='Per_Parameter')

    print("💾 Analisis kontekstual disimpan ke output/analisis_kontekstual.xlsx")

else:
    print("\n❌ DATA TIDAK SIAP. Proses perbandingan dibatalkan.")

print("\n--- SEMUA PROSES PERBANDINGAN AUTO-TUNING SELESAI ---")