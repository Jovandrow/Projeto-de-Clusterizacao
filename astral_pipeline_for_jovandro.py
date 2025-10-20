"""
Pipeline ASTRAL/SCOPe 2.08 – Projeto de Clusterização
Autor: Jovandro Wawrzonkiewicz Júnior
Professor: Aryel Marlus Repula de Oliveira
Matéria: Tópicos Especiais em Software
"""

import os, sys, argparse, itertools, gzip
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from joblib import Parallel, delayed

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    f1_score, adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, OPTICS,
    SpectralClustering, Birch, AffinityPropagation, MeanShift
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Aminoácidos padrão
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a:i for i,a in enumerate(AA)}
N_PARES = 400

# ------------------ LEITURA DO ARQUIVO ------------------
def ler_fasta(caminho):
    """Lê arquivo FASTA (txt ou gz) e retorna cabeçalhos e sequências"""
    openf = gzip.open if caminho.endswith('.gz') else open
    cabecalhos, seqs = [], []
    with openf(caminho, 'rt') as f:
        nome = None
        seq = []
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            if linha.startswith('>'):
                if nome:
                    cabecalhos.append(nome)
                    seqs.append(''.join(seq))
                nome = linha[1:].strip()
                seq = []
            else:
                seq.append(linha)
        if nome:
            cabecalhos.append(nome)
            seqs.append(''.join(seq))
    return cabecalhos, seqs

# ------------------ EXTRAÇÃO DE CLASSE ------------------
def extrair_classe(cabecalho):
    """Extrai a classe (ex: a.1.1.1) do cabeçalho SCOPe"""
    partes = cabecalho.split()
    for p in partes:
        if p.count('.') >= 2 and any(ch.isalpha() for ch in p):
            return p
    return partes[0]

# ------------------ K-MERS ------------------
def par_para_indice(par):
    """Converte um par de aminoácidos em índice numérico"""
    return AA_IDX[par[0]] * 20 + AA_IDX[par[1]]

def seq_para_pares(seq):
    """Transforma sequência em lista de pares de aminoácidos"""
    pares = []
    for i in range(len(seq)-1):
        a, b = seq[i].upper(), seq[i+1].upper()
        if a in AA_IDX and b in AA_IDX:
            pares.append(a+b)
        else:
            pares.append(None)
    return pares

def construir_matriz_esparsa(seqs, max_skip=0):
    """Cria matriz binária esparsa com presença/ausência de pares"""
    skips = list(range(0, max_skip+1))
    n_features = len(skips) * N_PARES * N_PARES 
    print(f"Gerando matriz com {len(seqs)} sequências e {n_features:,} características...")
    
    linhas, colunas, dados = [], [], []
    for si, seq in enumerate(tqdm(seqs, desc="Extraindo k-mers")):
        pares = seq_para_pares(seq)
        vistos = set()
        for i, p1 in enumerate(pares):
            if p1 is None: continue
            i1 = par_para_indice(p1)
            for sk in skips:
                j = i + 1 + sk
                if j < len(pares) and pares[j] is not None:
                    i2 = par_para_indice(pares[j])
                    
                    feat = sk * (N_PARES*N_PARES) + i1*N_PARES + i2 
                    
                    if feat not in vistos:
                        linhas.append(si)
                        colunas.append(feat)
                        dados.append(1)
                        vistos.add(feat)
                        
    X = sparse.csr_matrix((dados, (linhas, colunas)), shape=(len(seqs), n_features), dtype=np.uint8)
    return X

# ------------------ PCA (REDUÇÃO DE DIMENSÃO) ------------------
def aplicar_pca(X, n_comp=300):
    """Reduz a matriz binária com TruncatedSVD (versão para matrizes esparsas)"""
    print(f"Reduzindo para {n_comp} componentes principais...")
    svd = TruncatedSVD(n_components=min(n_comp, X.shape[1]-1), random_state=42)
    X_red = svd.fit_transform(X)
    return X_red, svd

# ------------------ MÉTRICAS ------------------
def mapear_clusters(preds, y_true):
    """Associa cada cluster ao rótulo mais comum dentro dele"""
    mapa = {}
    for c in np.unique(preds):
        idxs = np.where(preds == c)[0]
        if len(idxs) == 0: continue
        mais_comum = Counter(y_true[idxs]).most_common(1)[0][0]
        mapa[c] = mais_comum
    
    return np.array([mapa.get(c, -1) for c in preds])

def avaliar_cluster(nome, preds, X, y_true):
    """Calcula métricas internas e externas"""
    n_clusters = len(set(preds)) - (1 if -1 in preds else 0)
    resultado = {'algoritmo': nome, 'n_clusters': n_clusters}
    
    # Métricas Internas
    try:
        if n_clusters > 1 and n_clusters < X.shape[0]: 
            resultado['silhouette'] = silhouette_score(X, preds)
        else:
            resultado['silhouette'] = np.nan
    except:
        resultado['silhouette'] = np.nan
        
    try:
        resultado['calinski'] = calinski_harabasz_score(X, preds)
    except:
        resultado['calinski'] = np.nan
        
    try:
        resultado['davies'] = davies_bouldin_score(X, preds)
    except:
        resultado['davies'] = np.nan
        
    # Métricas Externas (F1-score é o principal objetivo)
    try:
        if n_clusters <= 1:
            raise ValueError("Não há clusters suficientes para F1-Score.")
            
        mapeado = mapear_clusters(preds, y_true)
        
        f1m = f1_score(y_true, mapeado, average='macro', zero_division=0)
        f1μ = f1_score(y_true, mapeado, average='micro', zero_division=0)
        resultado['f1_macro'] = f1m
        resultado['f1_micro'] = f1μ
        
        resultado['adj_rand_score'] = adjusted_rand_score(y_true, preds)
        resultado['nmi_score'] = normalized_mutual_info_score(y_true, preds)

    except Exception as e:
        # Em caso de falha (incluindo o caso de 1 cluster do DBSCAN), registra 0.0
        print(f"Aviso: Falha ao calcular F1/NMI para {nome}. Motivo: {e}")
        resultado['f1_macro'] = resultado['f1_micro'] = 0.0
        resultado['adj_rand_score'] = resultado['nmi_score'] = 0.0
        
    return resultado

# ------------------ CLUSTERIZAÇÃO ------------------
def rodar_clusterizacao(X, y_true, outdir="resultados"):
    """Executa diversos algoritmos de clusterização"""
    
    # CORREÇÃO/AJUSTE: Usa um valor fixo mais razoável para classes principais do SCOPe (5 a 10 classes principais)
    # Isso garante que os testes de K sejam mais relevantes
    n_classes_principais_fixo = 7 
    
    candidatos_k = [
        n_classes_principais_fixo,
        n_classes_principais_fixo + 1,
        max(2, n_classes_principais_fixo - 1),
        10, 20, 50
    ]
    candidatos_k = sorted(list(set(k for k in candidatos_k if k >= 2)))
    
    print(f"Executando clusterização com {len(y_true)} amostras, {len(np.unique(y_true))} classes totais e k's: {candidatos_k}")
    print("Iniciando testes de clusterização em paralelo...")

    def worker(nome, modelo, X, y_true):
        try:
            if hasattr(modelo, 'fit_predict'):
                preds = modelo.fit_predict(X)
            elif hasattr(modelo, 'fit') and hasattr(modelo, 'predict'):
                modelo.fit(X)
                preds = modelo.predict(X)
            else:
                 raise NotImplementedError("Modelo sem fit_predict ou fit/predict.")
                 
        except Exception as e:
            return {'algoritmo': nome, 'n_clusters': np.nan, 'silhouette': np.nan, 'calinski': np.nan, 
                    'davies': np.nan, 'f1_macro': 0.0, 'f1_micro': 0.0, 'adj_rand_score': 0.0, 'nmi_score': 0.0}
            
        return avaliar_cluster(nome, preds, X, y_true)

    jobs = []
    
    # TESTES DEPENDENTES DE K
    for k in candidatos_k:
        jobs.append(delayed(worker)(
            f"KMeans_k={k}", KMeans(n_clusters=k, random_state=42, n_init='auto'), X, y_true
        ))

        jobs.append(delayed(worker)(
            f"GMM_k={k}", GaussianMixture(n_components=k, random_state=42), X, y_true
        ))

        jobs.append(delayed(worker)(
            f"Agglomerativo_k={k}", AgglomerativeClustering(n_clusters=k, linkage='ward'), X, y_true
        ))

    # TESTES INDEPENDENTES DE K
    for eps in [0.5, 1.0, 2.0]:
        jobs.append(delayed(worker)(
            f"DBSCAN_eps={eps}", DBSCAN(eps=eps, min_samples=5), X, y_true
        ))
        
    jobs.append(delayed(worker)(
        "OPTICS", OPTICS(min_samples=5), X, y_true
    ))

    jobs.append(delayed(worker)(
        "Birch_k=20", Birch(n_clusters=20), X, y_true
    ))
    
    resultados = Parallel(n_jobs=-1)(jobs)
    
    df = pd.DataFrame(resultados)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "metricas_cluster.csv"), index=False)
    print("Resultados salvos em:", outdir)
    return df

# ------------------ PLOT ------------------
def grafico_correlacoes(df, outdir):
    """Cria gráfico de correlação entre métricas"""
    plt.figure(figsize=(8,6))
    metricas = df.select_dtypes(include=np.number).columns
    corr = df[metricas].corr()
    sns.heatmap(corr, annot=True, cmap="vlag", center=0)
    plt.title("Correlação entre métricas")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "correlacao_metricas.png"))
    plt.close()

# ------------------ MAIN ------------------
def main():
    parser = argparse.ArgumentParser(description="Pipeline ASTRAL/SCOPe em português")
    parser.add_argument("entrada", help="Arquivo FASTA (ex: arq.txt)")
    parser.add_argument("--max-skip", type=int, default=0, help="Número máximo de 'skips' entre pares (default=0)")
    parser.add_argument("--n-pca", type=int, default=300, help="Componentes principais do PCA")
    parser.add_argument("--saida", type=str, default="resultados", help="Pasta de saída")
    args = parser.parse_args()

    cabecalhos, seqs = ler_fasta(args.entrada)
    print(f"Lidas {len(seqs)} sequências do arquivo {args.entrada}")
    classes_str = [extrair_classe(c) for c in cabecalhos]
    print(f"Foram encontradas {len(set(classes_str))} classes únicas.")

    if len(seqs) < 2:
        print("Erro: Arquivo FASTA deve conter pelo menos duas sequências.")
        sys.exit(1)
        
    # CORREÇÃO: CODIFICAÇÃO DE RÓTULOS (Resolve o erro 'integer scalar arrays')
    encoder = LabelEncoder()
    classes_encoded = encoder.fit_transform(classes_str)
    
    X = construir_matriz_esparsa(seqs, args.max_skip)
    
    if X.shape[1] <= args.n_pca:
        print(f"Aviso: O número de features ({X.shape[1]}) é menor ou igual ao n_pca. Ajustando n_pca.")
        args.n_pca = X.shape[1] - 1
        
    X_pca, _ = aplicar_pca(X, args.n_pca)
    
    # Chamada da clusterização com as classes codificadas (classes_encoded)
    df = rodar_clusterizacao(X_pca, classes_encoded, args.saida)
    grafico_correlacoes(df, args.saida)
    
    print("\n" + "="*80)
    print("Pipeline finalizado com sucesso! ✅")
    
    # Impressão do melhor resultado
    melhor_modelo_df = df.sort_values(by='f1_macro', ascending=False).iloc[:1]
    
    print(f"\nMelhor Algoritmo (Criterio: F1-Macro):\n")
    
    # Imprime a linha do melhor modelo de forma limpa
    melhor_modelo_serie = melhor_modelo_df.iloc[0]
    
    # Formatação para remover o índice e o cabeçalho "Name: 0, dtype: object"
    output_string = melhor_modelo_serie.to_string()
    clean_output = output_string[output_string.find('\n')+1:].strip() 
    
    print(clean_output)
    
    print("="*80)

if __name__ == "__main__":
    main()
    