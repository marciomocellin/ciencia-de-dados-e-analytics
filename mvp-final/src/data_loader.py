"""
Módulo para carregamento e pré-processamento dos dados da Receita Federal (CNPJ)

Autor: MVP - Pós-Graduação em Ciência de Dados e Analytics - PUC-Rio
Data: Janeiro/2026
"""

import pandas as pd
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Layouts dos arquivos conforme documentação da Receita Federal
EMPRESAS_COLS = [
    'cnpj_basico', 'razao_social', 'natureza_juridica', 'qualificacao_responsavel',
    'capital_social', 'porte_empresa', 'ente_federativo_responsavel'
]

ESTABELECIMENTOS_COLS = [
    'cnpj_basico', 'cnpj_ordem', 'cnpj_dv', 'identificador_matriz_filial',
    'nome_fantasia', 'situacao_cadastral', 'data_situacao_cadastral',
    'motivo_situacao_cadastral', 'nome_cidade_exterior', 'pais',
    'data_inicio_atividade', 'cnae_fiscal_principal', 'cnae_fiscal_secundaria',
    'tipo_logradouro', 'logradouro', 'numero', 'complemento', 'bairro',
    'cep', 'uf', 'municipio', 'ddd_1', 'telefone_1', 'ddd_2', 'telefone_2',
    'ddd_fax', 'fax', 'correio_eletronico', 'situacao_especial',
    'data_situacao_especial'
]

SOCIOS_COLS = [
    'cnpj_basico', 'identificador_socio', 'nome_socio', 'cnpj_cpf_socio',
    'qualificacao_socio', 'data_entrada_sociedade', 'pais',
    'representante_legal', 'nome_representante', 'qualificacao_representante',
    'faixa_etaria'
]

PORTE_MAP = {
    '00': 'Não Informado',
    '01': 'Micro Empresa',
    '03': 'Empresa de Pequeno Porte',
    '05': 'Demais'
}

SITUACAO_MAP = {
    '01': 'Nula',
    '02': 'Ativa',
    '03': 'Suspensa',
    '04': 'Inapta',
    '08': 'Baixada'
}


def extract_zip_file(zip_path, extract_to):
    """
    Extrai arquivo ZIP
    
    Args:
        zip_path (Path): Caminho do arquivo ZIP
        extract_to (Path): Diretório de destino
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extraído: {zip_path.name}")


def load_csv_file(file_path, columns, encoding='latin1', sep=';', nrows=None):
    """
    Carrega arquivo CSV com tratamento de erros
    
    Args:
        file_path (Path): Caminho do arquivo
        columns (list): Lista de nomes das colunas
        encoding (str): Codificação do arquivo
        sep (str): Separador
        nrows (int): Número de linhas a carregar (None para todas)
        
    Returns:
        pd.DataFrame: DataFrame carregado
    """
    try:
        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding=encoding,
            names=columns,
            header=None,
            low_memory=False,
            nrows=nrows
        )
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar {file_path}: {e}")
        return None


def load_multiple_files(pattern, columns, data_dir, nrows=None):
    """
    Carrega múltiplos arquivos que seguem um padrão
    
    Args:
        pattern (str): Padrão glob para buscar arquivos
        columns (list): Lista de nomes das colunas
        data_dir (Path): Diretório onde buscar os arquivos
        nrows (int): Número de linhas a carregar por arquivo
        
    Returns:
        pd.DataFrame: DataFrame concatenado
    """
    dfs = []
    files = sorted(data_dir.glob(pattern))
    
    for file in files:
        print(f"Carregando: {file.name}")
        df = load_csv_file(file, columns, nrows=nrows)
        if df is not None:
            dfs.append(df)
    
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        print(f"✅ Total de registros: {len(result):,}")
        return result
    return None


def create_cnpj_completo(df):
    """
    Cria coluna com CNPJ completo formatado
    
    Args:
        df (pd.DataFrame): DataFrame com colunas cnpj_basico, cnpj_ordem, cnpj_dv
        
    Returns:
        pd.DataFrame: DataFrame com nova coluna cnpj_completo
    """
    df['cnpj_completo'] = (
        df['cnpj_basico'].astype(str).str.zfill(8) +
        df['cnpj_ordem'].astype(str).str.zfill(4) +
        df['cnpj_dv'].astype(str).str.zfill(2)
    )
    return df


def convert_dates(df, date_columns):
    """
    Converte colunas de data do formato YYYYMMDD para datetime
    
    Args:
        df (pd.DataFrame): DataFrame
        date_columns (list): Lista de colunas de data
        
    Returns:
        pd.DataFrame: DataFrame com datas convertidas
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
    return df


def calculate_company_age(df, date_col='data_inicio_atividade'):
    """
    Calcula idade da empresa em anos
    
    Args:
        df (pd.DataFrame): DataFrame com coluna de data de início
        date_col (str): Nome da coluna de data
        
    Returns:
        pd.DataFrame: DataFrame com nova coluna idade_empresa
    """
    df['idade_empresa'] = (pd.Timestamp.now() - df[date_col]).dt.days / 365.25
    return df


def map_descriptions(df, mapping_dict, from_col, to_col):
    """
    Mapeia códigos para descrições
    
    Args:
        df (pd.DataFrame): DataFrame principal
        mapping_dict (dict): Dicionário de mapeamento
        from_col (str): Coluna com códigos
        to_col (str): Nome da nova coluna com descrições
        
    Returns:
        pd.DataFrame: DataFrame com nova coluna
    """
    df[to_col] = df[from_col].astype(str).map(mapping_dict)
    return df


def preprocess_main_dataset(df_empresas, df_estabelecimentos, df_cnaes=None, 
                           df_municipios=None, df_naturezas=None):
    """
    Processa e junta os datasets principais
    
    Args:
        df_empresas (pd.DataFrame): DataFrame de empresas
        df_estabelecimentos (pd.DataFrame): DataFrame de estabelecimentos
        df_cnaes (pd.DataFrame): DataFrame de CNAEs (opcional)
        df_municipios (pd.DataFrame): DataFrame de municípios (opcional)
        df_naturezas (pd.DataFrame): DataFrame de naturezas jurídicas (opcional)
        
    Returns:
        pd.DataFrame: Dataset processado e consolidado
    """
    # Merge empresas + estabelecimentos
    df = df_estabelecimentos.merge(
        df_empresas,
        on='cnpj_basico',
        how='left',
        suffixes=('', '_empresa')
    )
    
    # Criar CNPJ completo
    df = create_cnpj_completo(df)
    
    # Converter datas
    df = convert_dates(df, ['data_situacao_cadastral', 'data_inicio_atividade'])
    
    # Calcular idade
    df = calculate_company_age(df)
    
    # Mapear descrições
    df = map_descriptions(df, PORTE_MAP, 'porte_empresa', 'porte_descricao')
    df = map_descriptions(df, SITUACAO_MAP, 'situacao_cadastral', 'situacao_descricao')
    
    # Juntar tabelas auxiliares
    if df_cnaes is not None:
        df = df.merge(
            df_cnaes.rename(columns={'codigo': 'cnae_fiscal_principal', 
                                    'descricao': 'cnae_descricao'}),
            on='cnae_fiscal_principal',
            how='left'
        )
    
    if df_municipios is not None:
        df = df.merge(
            df_municipios.rename(columns={'codigo': 'municipio', 
                                         'descricao': 'municipio_nome'}),
            on='municipio',
            how='left'
        )
    
    if df_naturezas is not None:
        df = df.merge(
            df_naturezas.rename(columns={'codigo': 'natureza_juridica', 
                                        'descricao': 'natureza_descricao'}),
            on='natureza_juridica',
            how='left'
        )
    
    return df


def load_all_data(data_raw_dir, extract_dir, sample_size=None):
    """
    Carrega todos os dados necessários para análise
    
    Args:
        data_raw_dir (Path): Diretório com arquivos ZIP
        extract_dir (Path): Diretório para extração
        sample_size (int): Tamanho da amostra (None para carregar tudo)
        
    Returns:
        dict: Dicionário com todos os DataFrames carregados
    """
    # Criar diretório de extração
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Extrair ZIPs se necessário
    if not list(extract_dir.glob('*.csv')):
        print("Extraindo arquivos ZIP...")
        for zip_file in list(data_raw_dir.glob('*.zip'))[:5]:
            extract_zip_file(zip_file, extract_dir)
    
    # Carregar dados
    print("\n" + "="*60)
    print("CARREGANDO DADOS")
    print("="*60)
    
    datasets = {}
    
    print("\nEmpresas:")
    datasets['empresas'] = load_multiple_files(
        '*.EMPRECSV', EMPRESAS_COLS, extract_dir, nrows=sample_size
    )
    
    print("\nEstabelecimentos:")
    datasets['estabelecimentos'] = load_multiple_files(
        '*.ESTABELE', ESTABELECIMENTOS_COLS, extract_dir, nrows=sample_size
    )
    
    print("\nSócios:")
    datasets['socios'] = load_multiple_files(
        '*.SOCIOCSV', SOCIOS_COLS, extract_dir, nrows=sample_size
    )
    
    # Tabelas auxiliares
    print("\nTabelas auxiliares:")
    
    cnae_files = list(extract_dir.glob('*.CNAECSV'))
    if cnae_files:
        datasets['cnaes'] = load_csv_file(cnae_files[0], ['codigo', 'descricao'])
    
    muni_files = list(extract_dir.glob('*.MUNICCSV'))
    if muni_files:
        datasets['municipios'] = load_csv_file(muni_files[0], ['codigo', 'descricao'])
    
    nat_files = list(extract_dir.glob('*.NATJUCSV'))
    if nat_files:
        datasets['naturezas'] = load_csv_file(nat_files[0], ['codigo', 'descricao'])
    
    qual_files = list(extract_dir.glob('*.QUALSCSV'))
    if qual_files:
        datasets['qualificacoes'] = load_csv_file(qual_files[0], ['codigo', 'descricao'])
    
    return datasets
