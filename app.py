# Importando as bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
from scikitlearn.tree import DecisionTreeClassifier
from scikitlearn.model_selection import train_test_split
from scikitlearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scikitlearn.tree import plot_tree

# Definindo o título do app
st.title("Prevendo Diabetes")

# Criando um botão de upload
uploaded_file = st.file_uploader("Faça upload do arquivo csv com os dados da Kaggle", type="csv")

# Verificando se o arquivo foi carregado
if uploaded_file is not None:
    # Lendo o arquivo csv
    df = pd.read_csv(uploaded_file)
    # Mostrando as primeiras linhas do dataframe
    st.write(df.head())
else:
    # Mostrando uma mensagem de erro se o arquivo não foi carregado
    st.info("""Ainda não possui o arquivo?
    [Clique Aqui](https://www.kaggle.com/uciml/pima-indians-diabetes-database/download) para fazer download.
    O arquivo vem compactado, lembre-se de extrair em alguma pasta de sua preferência.""")

# Criando um expander para mostrar mais informações sobre a diabetes
expander = st.expander("Saiba mais sobre a diabetes")
expander.write("""
A diabetes é uma doença crônica que afeta a forma como o corpo metaboliza a glicose, a principal fonte de energia do organismo. A glicose é obtida a partir dos alimentos que ingerimos e é transportada pelo sangue até as células, onde é utilizada como combustível. Para que isso aconteça, é necessário que o pâncreas produza um hormônio chamado insulina, que facilita a entrada da glicose nas células. Quando o pâncreas não produz insulina suficiente ou quando o organismo não consegue usar adequadamente a insulina que produz, ocorre um aumento da glicose no sangue, chamado de hiperglicemia. A hiperglicemia crônica pode causar danos em diversos órgãos, como olhos, rins, nervos, coração e vasos sanguíneos.

Existem dois tipos principais de diabetes: o tipo 1 e o tipo 2. O diabetes tipo 1 é uma doença autoimune, que ocorre quando o sistema imunológico ataca as células produtoras de insulina do pâncreas, destruindo-as ou inibindo sua função. O diabetes tipo 1 geralmente se manifesta na infância ou na adolescência, mas pode ocorrer em qualquer idade. As pessoas com diabetes tipo 1 precisam tomar insulina diariamente para controlar a glicose no sangue. O diabetes tipo 2 é o mais comum e está relacionado à resistência à ação da insulina, ou seja, o organismo produz insulina, mas não consegue usá-la de forma eficaz. O diabetes tipo 2 está associado a fatores de risco como obesidade, sedentarismo, histórico familiar, idade avançada e etnia. O diabetes tipo 2 pode ser controlado com medicamentos orais ou injetáveis, além de mudanças no estilo de vida, como alimentação saudável e atividade física.

Os sintomas da diabetes podem variar de acordo com o tipo, a duração e a gravidade da doença. Alguns dos sintomas mais comuns são: sede excessiva, fome excessiva, micção frequente, perda de peso, cansaço, visão embaçada, infecções recorrentes, feridas que demoram a cicatrizar, formigamento ou dormência nas mãos ou nos pés. É importante ressaltar que nem todas as pessoas com diabetes apresentam sintomas, especialmente no caso do diabetes tipo 2, que pode permanecer silencioso por anos. Por isso, é recomendado fazer exames periódicos de glicemia, especialmente se houver fatores de risco.

O diagnóstico da diabetes é feito com base em exames de sangue que medem a glicose no sangue em jejum ou após uma sobrecarga de glicose. Os critérios para o diagnóstico são: glicemia em jejum maior ou igual a 126 mg/dL em duas ocasiões diferentes; glicemia maior ou igual a 200 mg/dL duas horas após a ingestão de 75 g de glicose; glicemia maior ou igual a 200 mg/dL em qualquer momento do dia, associada a sintomas de diabetes; hemoglobina glicada (A1C) maior ou igual a 6,5% em duas ocasiões diferentes. A hemoglobina glicada é um exame que reflete a média da glicose no sangue nos últimos três meses.

O tratamento da diabetes visa manter a glicose no sangue dentro dos valores normais, evitando as complicações da doença. O tratamento envolve o uso de medicamentos, a prática de atividade física, o controle do peso corporal, a alimentação saudável e o monitoramento da glicose no sangue. O tratamento deve ser individualizado e acompanhado por uma equipe multidisciplinar, composta por médico, enfermeiro, nutricionista, educador físico, psicólogo, entre outros profissionais. O autocuidado é fundamental para o sucesso do tratamento, ou seja, a pessoa com diabetes deve se responsabilizar pelo seu próprio cuidado, seguindo as orientações da equipe de saúde e participando ativamente das decisões sobre o seu tratamento.

A prevenção da diabetes é possível, principalmente no caso do diabetes tipo 2. A prevenção envolve a adoção de hábitos de vida saudáveis, como evitar o tabagismo, o consumo excessivo de álcool, o estresse, o sedentarismo e a obesidade. A alimentação saudável é um dos pilares da prevenção, pois ajuda a controlar o peso corporal, a glicose no sangue, o colesterol e a pressão arterial. A alimentação saudável deve ser equilibrada, variada, colorida e adequada às necessidades de cada pessoa. Alguns dos alimentos que devem ser consumidos com moderação são: açúcar, doces, refrigerantes, sucos industrializados, farinha branca, pão branco, arroz branco, massas, batata, mandioca, frituras, gorduras saturadas, gorduras trans, sal e alimentos processados. Alguns dos alimentos que devem ser consumidos com frequência são: frutas, verduras, legumes, cereais integrais, leguminosas, oleaginosas, sementes, leite e derivados desnatados, carnes magras, peixes, ovos, azeite de oliva, alho, cebola, ervas e especiarias.

Fontes: [Sociedade Brasileira de Diabetes](http://www2.datasus.gov.br/SIAB/index.php?area=02), [Ministério da Saúde](https://www.kaggle.com/datasets/datahackers/state-of-data-2021), [Portal Data Science](https://portaldatascience.com/kaggle/)""")
# Criando um sidebar para inserir as informações do usuário
st.sidebar.header("Insira as informações do paciente:")
# Criando um seletor para o sexo
sex = st.sidebar.selectbox("Sexo", ("Masculino", "Feminino"))
# Convertendo o sexo em binário
if sex == "Masculino":
    sex = 1
else:
    sex = 0
# Criando um slider para a idade
age = st.sidebar.slider("Idade", 0, 100, 25)
# Criando um slider para o número de gestações
preg = st.sidebar.slider("Número de gestações", 0, 20, 0)
# Criando um slider para a glicose
glu = st.sidebar.slider("Glicose", 0, 200, 100)
# Criando um slider para a pressão arterial
bp = st.sidebar.slider("Pressão arterial", 0, 120, 60)
# Criando um slider para a espessura da pele
skin = st.sidebar.slider("Espessura da pele", 0, 100, 20)
# Criando um slider para a insulina
ins = st.sidebar.slider("Insulina", 0, 800, 100)
# Criando um slider para o índice de massa corporal
bmi = st.sidebar.slider("Índice de massa corporal", 0, 50, 25)
# Criando um slider para a hereditariedade
ped = st.sidebar.slider("Hereditariedade", 0.0, 1.0, 0.5)
# Criando um botão para fazer a previsão
button = st.sidebar.button("Fazer a previsão")

# Criando um dataframe com as informações do usuário
user_data = pd.DataFrame({
    "Pregnancies": [preg],
    "Glucose": [glu],
    "BloodPressure": [bp],
    "SkinThickness": [skin],
    "Insulin": [ins],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [ped],
    "Age": [age],
    "Outcome": [0]
})

# Verificando se o botão foi clicado
if button:
    # Separando as variáveis independentes e dependentes
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Treinando um classificador de árvore de decisão
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Fazendo previsões para os dados de teste
    y_pred = clf.predict(X_test)

    # Calculando a acurácia das previsões
    acc = accuracy_score(y_test, y_pred)
    st.write(f"A acurácia do modelo é {acc:.2f}")

    # Exibindo a árvore de decisão
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True, ax=ax)
    st.pyplot(fig)

    # Fazendo a previsão para o usuário
    user_pred = clf.predict(user_data.drop("Outcome", axis=1))
    if user_pred[0] == 0:
        st.success("Parabéns! Você não tem diabetes.")
    else:
        st.error("Atenção! Você tem diabetes.")

# Inserindo um aviso informando que é um modelo de teste
st.warning("Atenção: este aplicativo é um modelo de teste e não substitui um diagnóstico médico profissional. Consulte um médico se você tiver sintomas ou suspeita de diabetes.")