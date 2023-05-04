# Definir función para obtener datos que coinciden con una consulta
def get_data_matching_query(df, query):
    """
    Filtra un DataFrame según las condiciones especificadas en la consulta.

    Args:
    df (pd.DataFrame): DataFrame original que se desea filtrar.
    query (list): Lista de condiciones de filtrado. Cada elemento es una columna en el DataFrame original
                  y se filtran las filas en las que el valor de esa columna es igual a 1.

    Returns:
    query_df (pd.DataFrame): DataFrame filtrado según las condiciones especificadas en la consulta.
    """
    query_df = df.copy()
    for q in query:
        query_df = query_df[query_df[q] == 1]
    return query_df


def plot_confusion_matrix(cm, model_name, target):
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False, annot_kws={"fontsize": 12})
    plt.title(f'Matriz de Confusión para {model_name} ({target})', fontsize=14)
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.ylabel('Clase Real', fontsize=12)
    plt.show()

def plot_auc_roc(y_test, y_proba, model_name, target):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'Curva AUC ROC para {model_name} ({target})', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    plt.show()