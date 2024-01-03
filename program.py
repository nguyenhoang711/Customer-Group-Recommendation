from data_visualizing import *
from data_preprocessing import *

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.tree import export_graphviz
from IPython.display import Image
from six import StringIO
import pydotplus


def read_file(fileName):
    try:
        data = pd.read_csv(fileName)
        return data
    except FileNotFoundError:
        print(f"Không tìm thấy file ${fileName} !")


def create_tree_image(clf, data, image_name):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,feature_names =data.columns[:7],
                    class_names=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(image_name)
    Image(graph.create_png())


mall = read_file('data/segmentation_data.csv')
print(mall.head())
#-------------------------------- TIỀN XỬ LÝ DỮ LIỆU -------------------------------------
check_and_drop_duplicated(mall)
check_null(mall)
unnecess_cols = ['ID']
mall = remove_unnecess_cols(mall, unnecess_cols)
log_transformed_age = apply_log(mall['Age'])
mall['transf_age'] = log_transformed_age
mall['transf_income'] = power_transform(mall['Income'])
duplicated_cols = ['Income', 'Age']
mall_transformed = remove_unnecess_cols(mall, duplicated_cols)
#-------------------------------- CHUẨN HÓA DỮ LIỆU -------------------------------------
mall_norm = min_max_scaling(mall_transformed)
mall_norm = pd.DataFrame(data = mall_norm,columns = mall_transformed.columns)
print(mall_norm.head())
#--------------------------------- Tạo mô hình K-means Clustering ------------------------
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
kmeans.fit(mall_norm)
mall_segm_kmeans= mall_norm.copy()
mall_segm_kmeans['SegmentKmeans'] = kmeans.labels_
mall_analysis = mall_segm_kmeans.groupby(['SegmentKmeans']).mean()
#--------------------------------- Giảm chiều dữ liệu PCA kết hợp Kmeans -------------------
mall_kmeans = remove_unnecess_cols(mall_segm_kmeans, 'SegmentKmeans')
pca = PCA(n_components = 3)
pca.fit(mall_kmeans)
cols = ['Component 1', 'Component 2', 'Component 3']
df_pca_comp = pd.DataFrame(data = pca.components_,
                           columns = mall_kmeans.columns,
              index = cols)
title = 'Components vs Original Features'
min_range = -1
max_range = 1
create_heatmap(df_pca_comp, min_range, max_range, cols, title)
scores_pca = pca.transform(mall_kmeans)
kmeans_pca = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
kmeans_pca.fit(scores_pca)
results_df = pd.DataFrame(scores_pca, columns=['Component 1', 'Component 2', 'Component 3'])
results_df['Labels'] = kmeans_pca.labels_
x_axis = 'Component 2'
y_axis = 'Component 1'
label_column = 'Labels'
title2 = 'Clusters by PCA Components'
create_scatter(results_df, x_axis, y_axis,label_column, title2)
#---------------------------------- đưa về bộ dữ liệu ban đầu kết hợp với nhãn của cụm ----------------------
results_info = mall.drop(['transf_income', 'transf_age'], axis=1)
results_info['Labels'] = kmeans.labels_
results_info = results_info.astype({'Sex':'int32', 'Marital status':'int32', 'Education':'int32', 'Occupation':'int32', 'Settlement size':'int32'})
results_info.info()
#-------------------------------- tao mô hình cây phân cụm cho Kmeans ----------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 5)
X_clusters = results_info.drop('Labels', axis=1)
y_clusters = results_info['Labels']

clf.fit(X_clusters, y_clusters)
#--------------------------------- Đánh giá mô hình phân cụm ------------------------------------------------------
from sklearn.metrics import classification_report
predictions = clf.predict(X_clusters)
print(classification_report(y_clusters, predictions))
create_tree_image(clf, results_info, "output/DecisionTree.png")