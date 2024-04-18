import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def load_data():
    url = "Data Cleaned.csv"
    df = pd.read_csv(url)
    return df

def get_top_10_heroes(df):
    # Menghitung skor gabungan
    df['Score'] = df['T_Picked'] * df['T_WinRate'] * df['T_PickPercentage'] / (df['T_Loses'] + 1)

    # Mengurutkan berdasarkan skor dan mengambil top 10
    top_10_heroes = df.nlargest(10, 'Score')
    return top_10_heroes

def main():
    
    st.title('Analisis Performa Hero dalam kejuaraan M5 World Championship')

    st.markdown("---")

    # Load data
    df = load_data()
    st.write('Dataset: ')
    st.write(df)

    st.markdown("---")

    role_pick_percentage = df.groupby('Roles')['T_PickPercentage'].sum()

    st.subheader('Total Pick Percentage Berdasarkan Role')
    fig_donut = px.pie(names=role_pick_percentage.index.tolist(), values=role_pick_percentage.values.tolist(), hole=0.2)
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(width=460, height=460)
    st.plotly_chart(fig_donut)
    st.write("Grafik donat di atas menampilkan total pick presentase berdasarkan peran (Roles) dalam dataset. Setiap bagian donat mewakili persentase dari total pick presentase dalam masing-masing peran.")
    st.markdown("---")

   

    role_ban_percentage = df.groupby('Roles')['T_BansPercentage'].sum()

    st.subheader('Total Ban Percentage Berdasarkan Role')
    fig_donut = px.pie(names=role_ban_percentage.index.tolist(), values=role_ban_percentage.values.tolist(), hole=0.2)
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(width=460, height=460)
    st.plotly_chart(fig_donut)
    st.write("Grafik donat di atas menampilkan total ban presentase berdasarkan peran (Roles) dalam dataset. Setiap bagian donat mewakili persentase dari total ban presentase dalam masing-masing peran.")
    st.markdown("---")

    # Visualisasi top 10 hero berdasarkan kriteria gabungan
    top_10_heroes = get_top_10_heroes(df)
    st.subheader('Top 10 Hero Berdasarkan Kriteria Gabungan')
    fig = px.bar(top_10_heroes, x='Score', y='Hero', orientation='h', 
                 title='Top 10 Hero Berdasarkan Kriteria Gabungan',
                 labels={'Score': 'Score', 'Hero': 'Hero Name'})
    fig.update_xaxes(title='Score')
    fig.update_yaxes(title='Hero Name')
    st.plotly_chart(fig)

    st.markdown("---")

    st.subheader('Jumlah Hero Berdasarkan Role')
    role_counts = df['Roles'].value_counts()
    st.bar_chart(role_counts)

    st.subheader('Hubungan antara Total Win dan Total Lose dengan Total Win Rate')
    fig = px.scatter(df, x='T_WinRate', y='T_Loses', color='Roles', hover_name='Hero', size='T_Wins')
    fig.update_layout(xaxis_title='Total Win Rate', yaxis_title='Total Loses', title='Hubungan antara Total Win dan Total Lose dengan Total Win Rate')
    st.plotly_chart(fig)



    st.markdown("---")

    st.sidebar.header('Pilih Hero')
    df['Nama dan Role Hero'] = df['Hero'] + ' (' + df['Roles'] + ')'
    option = st.sidebar.selectbox('Nama dan Role Hero', df['Nama dan Role Hero'])

    X_train_numeric = df.drop(['Hero', 'Roles', 'Nama dan Role Hero'], axis=1)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X_train_numeric)

    # Compute silhouette scores for different number of clusters
    silhouette_scores = []
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Visualize Silhouette Score
    st.subheader('Silhouette Score')
    fig_silhouette = px.line(x=range(2, 11), y=silhouette_scores, markers=True)
    fig_silhouette.update_layout(xaxis_title='Number of Clusters', yaxis_title='Silhouette Score')
    st.plotly_chart(fig_silhouette)
    st.write("Grafik di atas menunjukkan Skor Silhouette untuk jumlah cluster yang berbeda. Skor Silhouette mengukur seberapa mirip suatu objek dengan clusternya sendiri (kohesi) dibandingkan dengan cluster lain (pemisahan). Skor Siluet yang lebih tinggi menunjukkan pembentukan cluster yang lebih baik.")
    st.markdown("---")

    # Model K-Means Clustering
    st.subheader('K-Means Clustering Model')


    # # Buat objek model HAC dengan jumlah cluster yang dipilih
    # Choose the number of clusters based on the highest silhouette score
    n_clusters_kmeans = silhouette_scores.index(max(silhouette_scores)) + 2

    # Create KMeans model
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)

    # Fit the model to the standardized features
    cluster_labels_kmeans = kmeans.fit_predict(scaled_features)

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(scaled_features)

    # Create a DataFrame for visualization
    scatter_data_kmeans = pd.DataFrame({
        'Principal Component 1': X_train_pca[:, 0],
        'Principal Component 2': X_train_pca[:, 1],
        'Cluster': cluster_labels_kmeans
    })

    # Visualize using Plotly
    fig_kmeans = px.scatter(scatter_data_kmeans, x='Principal Component 1', y='Principal Component 2', color='Cluster',
                            title='Visualization of K-Means Clustering', 
                            labels={'Principal Component 1': 'Principal Component 1', 'Principal Component 2': 'Principal Component 2'},
                            hover_name=scatter_data_kmeans.index,
                            color_continuous_scale=['red', 'white'])

    st.plotly_chart(fig_kmeans)
    st.write("Diagram di atas memvisualisasikan hasil pengelompokan menggunakan PCA. Setiap titik mewakili Hero-Hero yang ada dan diwarnai berdasarkan kluster mereka.")


    if st.sidebar.button('Rekomendasikan Hero Serupa Berdasarkan Total Percentage Pick Dan Total Win Rate Hero '):
        
        # Fungsi untuk merekomendasikan hero dari role yang sama
        def recommend_similar_heroes_by_role(input_hero, df, cluster_labels, input_hero_cluster, n_recommendations=5):
            input_hero_index = df.index[df['Nama dan Role Hero'] == input_hero][0]
            input_hero_role = df.iloc[input_hero_index]['Roles']
            input_hero_win_rate = df.iloc[input_hero_index]['T_WinRate']  # Total Win Rate Percentage
            input_hero_pick_rate = df.iloc[input_hero_index]['T_PickPercentage']  # Total Pick Percentage
            
            similar_heroes_indices = [i for i, label in enumerate(cluster_labels) if 
                                    label == input_hero_cluster and i != input_hero_index and 
                                    df.iloc[i]['Roles'] == input_hero_role]
            similar_heroes = df.iloc[similar_heroes_indices]

            # Calculate the distance or similarity based on win rate and pick rate
            similar_heroes['WinRateDifference'] = abs(similar_heroes['T_WinRate'] - input_hero_win_rate)
            similar_heroes['PickRateDifference'] = abs(similar_heroes['T_PickPercentage'] - input_hero_pick_rate)

            # Combine the differences to get a total score (you can adjust weights if needed)
            similar_heroes['TotalDifference'] = similar_heroes['WinRateDifference'] + similar_heroes['PickRateDifference']

            # Sort the similar heroes by their total difference (lower is better)
            similar_heroes = similar_heroes.sort_values(by='TotalDifference', ascending=True)

            recommended_heroes = similar_heroes.head(n_recommendations)
            recommended_titles = recommended_heroes['Nama dan Role Hero'].tolist()

            return recommended_titles

        input_hero_index = df.index[df['Nama dan Role Hero'] == option][0]
        input_hero_cluster = cluster_labels_kmeans[input_hero_index]
        recommended_heroes_by_role = recommend_similar_heroes_by_role(option, df, cluster_labels_kmeans, input_hero_cluster)
        for hero in recommended_heroes_by_role:
            st.sidebar.write(hero)

    # Tampilkan widget untuk memilih kluster
    selected_cluster = st.sidebar.selectbox('Pilih Kluster', sorted(set(cluster_labels_kmeans)))

    # Fungsi untuk merekomendasikan hero dari kluster tertentu
    def recommend_heroes_by_cluster(df, cluster_labels, selected_cluster, n_recommendations=110):
        heroes_in_cluster = df[cluster_labels == selected_cluster]['Nama dan Role Hero'].tolist()
        return heroes_in_cluster[:n_recommendations]

    # Tampilkan rekomendasi hero berdasarkan kluster yang dipilih
    recommended_heroes_selected_cluster = recommend_heroes_by_cluster(df, cluster_labels_kmeans, selected_cluster)
    st.sidebar.subheader(f'Hero di Kluster {selected_cluster}:')
    for hero in recommended_heroes_selected_cluster:
        st.sidebar.write(hero)




if __name__ == "__main__":
    main()
