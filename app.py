# Importando as bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from io import BytesIO
import base64

import streamlit as st

tab1, tab2, tab3 = st.tabs(["Uso do App", "Sobre a Diabetes", "Métricas adicionais"])

with tab1:
   st.header("Funcionalidades do App.")
   st.write(""" Este aplicativo oferece a capacidade de prever a ocorrência de diabetes com base em dados fornecidos. Você pode carregar um arquivo CSV contendo informações médicas, inserir dados de um paciente e fazer previsões sobre a presença de diabetes. Após inserir as informações do paciente, clique no botão 'Fazer a previsão' para obter o diagnóstico.""")

with tab2:
   st.header("Sobre a diabetes.")
   st.write(""" A diabetes é uma doença crônica que afeta a forma como o corpo metaboliza a glicose, a principal fonte de energia do organismo. A glicose é obtida a partir dos alimentos que ingerimos e é transportada pelo sangue até as células, onde é utilizada como combustível. Para que isso aconteça, é necessário que o pâncreas produza um hormônio chamado insulina, que facilita a entrada da glicose nas células. Quando o pâncreas não produz insulina suficiente ou quando o organismo não consegue usar adequadamente a insulina que produz, ocorre um aumento da glicose no sangue, chamado de hiperglicemia. A hiperglicemia crônica pode causar danos em diversos órgãos, como olhos, rins, nervos, coração e vasos sanguíneos.

Existem dois tipos principais de diabetes: o tipo 1 e o tipo 2. O diabetes tipo 1 é uma doença autoimune, que ocorre quando o sistema imunológico ataca as células produtoras de insulina do pâncreas, destruindo-as ou inibindo sua função. O diabetes tipo 1 geralmente se manifesta na infância ou na adolescência, mas pode ocorrer em qualquer idade. As pessoas com diabetes tipo 1 precisam tomar insulina diariamente para controlar a glicose no sangue. O diabetes tipo 2 é o mais comum e está relacionado à resistência à ação da insulina, ou seja, o organismo produz insulina, mas não consegue usá-la de forma eficaz. O diabetes tipo 2 está associado a fatores de risco como obesidade, sedentarismo, histórico familiar, idade avançada e etnia. O diabetes tipo 2 pode ser controlado com medicamentos orais ou injetáveis, além de mudanças no estilo de vida, como alimentação saudável e atividade física.

Os sintomas da diabetes podem variar de acordo com o tipo, a duração e a gravidade da doença. Alguns dos sintomas mais comuns são: sede excessiva, fome excessiva, micção frequente, perda de peso, cansaço, visão embaçada, infecções recorrentes, feridas que demoram a cicatrizar, formigamento ou dormência nas mãos ou nos pés. É importante ressaltar que nem todas as pessoas com diabetes apresentam sintomas, especialmente no caso do diabetes tipo 2, que pode permanecer silencioso por anos. Por isso, é recomendado fazer exames periódicos de glicemia, especialmente se houver fatores de risco.

O diagnóstico da diabetes é feito com base em exames de sangue que medem a glicose no sangue em jejum ou após uma sobrecarga de glicose. Os critérios para o diagnóstico são: glicemia em jejum maior ou igual a 126 mg/dL em duas ocasiões diferentes; glicemia maior ou igual a 200 mg/dL duas horas após a ingestão de 75 g de glicose; glicemia maior ou igual a 200 mg/dL em qualquer momento do dia, associada a sintomas de diabetes; hemoglobina glicada (A1C) maior ou igual a 6,5% em duas ocasiões diferentes. A hemoglobina glicada é um exame que reflete a média da glicose no sangue nos últimos três meses.

O tratamento da diabetes visa manter a glicose no sangue dentro dos valores normais, evitando as complicações da doença. O tratamento envolve o uso de medicamentos, a prática de atividade física, o controle do peso corporal, a alimentação saudável e o monitoramento da glicose no sangue. O tratamento deve ser individualizado e acompanhado por uma equipe multidisciplinar, composta por médico, enfermeiro, nutricionista, educador físico, psicólogo, entre outros profissionais. O autocuidado é fundamental para o sucesso do tratamento, ou seja, a pessoa com diabetes deve se responsabilizar pelo seu próprio cuidado, seguindo as orientações da equipe de saúde e participando ativamente das decisões sobre o seu tratamento.

A prevenção da diabetes é possível, principalmente no caso do diabetes tipo 2. A prevenção envolve a adoção de hábitos de vida saudáveis, como evitar o tabagismo, o consumo excessivo de álcool, o estresse, o sedentarismo e a obesidade. A alimentação saudável é um dos pilares da prevenção, pois ajuda a controlar o peso corporal, a glicose no sangue, o colesterol e a pressão arterial. A alimentação saudável deve ser equilibrada, variada, colorida e adequada às necessidades de cada pessoa. Alguns dos alimentos que devem ser consumidos com moderação são: açúcar, doces, refrigerantes, sucos industrializados, farinha branca, pão branco, arroz branco, massas, batata, mandioca, frituras, gorduras saturadas, gorduras trans, sal e alimentos processados. Alguns dos alimentos que devem ser consumidos com frequência são: frutas, verduras, legumes, cereais integrais, leguminosas, oleaginosas, sementes, leite e derivados desnatados, carnes magras, peixes, ovos, azeite de oliva, alho, cebola, ervas e especiarias.

Fontes: [Sociedade Brasileira de Diabetes](http://www2.datasus.gov.br/SIAB/index.php?area=02), [Ministério da Saúde](https://www.kaggle.com/datasets/datahackers/state-of-data-2021), [Portal Data Science](https://portaldatascience.com/kaggle/)""")

with tab3:
   st.header("Métricas adicionais")
   st.image("Teste")
# Definindo o título do app
st.title("Prevendo Diabetes")
st.divider()
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
expander_funcionalidades = st.expander("ℹ️ Funcionalidades do App")
expander_funcionalidades.write("Este aplicativo oferece a capacidade de prever a ocorrência de diabetes com base em dados fornecidos.")
expander_funcionalidades.write("Você pode carregar um arquivo CSV contendo informações médicas, inserir dados de um paciente e fazer previsões sobre a presença de diabetes.")
expander_funcionalidades.write("Após inserir as informações do paciente, clique no botão 'Fazer a previsão' para obter o diagnóstico.")

expander = st.expander("Saiba mais sobre a diabetes")
expander.write("""
A diabetes é uma doença crônica que afeta a forma como o corpo metaboliza a glicose, a principal fonte de energia do organismo. A glicose é obtida a partir dos alimentos que ingerimos e é transportada pelo sangue até as células, onde é utilizada como combustível. Para que isso aconteça, é necessário que o pâncreas produza um hormônio chamado insulina, que facilita a entrada da glicose nas células. Quando o pâncreas não produz insulina suficiente ou quando o organismo não consegue usar adequadamente a insulina que produz, ocorre um aumento da glicose no sangue, chamado de hiperglicemia. A hiperglicemia crônica pode causar danos em diversos órgãos, como olhos, rins, nervos, coração e vasos sanguíneos.

Existem dois tipos principais de diabetes: o tipo 1 e o tipo 2. O diabetes tipo 1 é uma doença autoimune, que ocorre quando o sistema imunológico ataca as células produtoras de insulina do pâncreas, destruindo-as ou inibindo sua função. O diabetes tipo 1 geralmente se manifesta na infância ou na adolescência, mas pode ocorrer em qualquer idade. As pessoas com diabetes tipo 1 precisam tomar insulina diariamente para controlar a glicose no sangue. O diabetes tipo 2 é o mais comum e está relacionado à resistência à ação da insulina, ou seja, o organismo produz insulina, mas não consegue usá-la de forma eficaz. O diabetes tipo 2 está associado a fatores de risco como obesidade, sedentarismo, histórico familiar, idade avançada e etnia. O diabetes tipo 2 pode ser controlado com medicamentos orais ou injetáveis, além de mudanças no estilo de vida, como alimentação saudável e atividade física.

Os sintomas da diabetes podem variar de acordo com o tipo, a duração e a gravidade da doença. Alguns dos sintomas mais comuns são: sede excessiva, fome excessiva, micção frequente, perda de peso, cansaço, visão embaçada, infecções recorrentes, feridas que demoram a cicatrizar, formigamento ou dormência nas mãos ou nos pés. É importante ressaltar que nem todas as pessoas com diabetes apresentam sintomas, especialmente no caso do diabetes tipo 2, que pode permanecer silencioso por anos. Por isso, é recomendado fazer exames periódicos de glicemia, especialmente se houver fatores de risco.

O diagnóstico da diabetes é feito com base em exames de sangue que medem a glicose no sangue em jejum ou após uma sobrecarga de glicose. Os critérios para o diagnóstico são: glicemia em jejum maior ou igual a 126 mg/dL em duas ocasiões diferentes; glicemia maior ou igual a 200 mg/dL duas horas após a ingestão de 75 g de glicose; glicemia maior ou igual a 200 mg/dL em qualquer momento do dia, associada a sintomas de diabetes; hemoglobina glicada (A1C) maior ou igual a 6,5% em duas ocasiões diferentes. A hemoglobina glicada é um exame que reflete a média da glicose no sangue nos últimos três meses.

O tratamento da diabetes visa manter a glicose no sangue dentro dos valores normais, evitando as complicações da doença. O tratamento envolve o uso de medicamentos, a prática de atividade física, o controle do peso corporal, a alimentação saudável e o monitoramento da glicose no sangue. O tratamento deve ser individualizado e acompanhado por uma equipe multidisciplinar, composta por médico, enfermeiro, nutricionista, educador físico, psicólogo, entre outros profissionais. O autocuidado é fundamental para o sucesso do tratamento, ou seja, a pessoa com diabetes deve se responsabilizar pelo seu próprio cuidado, seguindo as orientações da equipe de saúde e participando ativamente das decisões sobre o seu tratamento.

A prevenção da diabetes é possível, principalmente no caso do diabetes tipo 2. A prevenção envolve a adoção de hábitos de vida saudáveis, como evitar o tabagismo, o consumo excessivo de álcool, o estresse, o sedentarismo e a obesidade. A alimentação saudável é um dos pilares da prevenção, pois ajuda a controlar o peso corporal, a glicose no sangue, o colesterol e a pressão arterial. A alimentação saudável deve ser equilibrada, variada, colorida e adequada às necessidades de cada pessoa. Alguns dos alimentos que devem ser consumidos com moderação são: açúcar, doces, refrigerantes, sucos industrializados, farinha branca, pão branco, arroz branco, massas, batata, mandioca, frituras, gorduras saturadas, gorduras trans, sal e alimentos processados. Alguns dos alimentos que devem ser consumidos com frequência são: frutas, verduras, legumes, cereais integrais, leguminosas, oleaginosas, sementes, leite e derivados desnatados, carnes magras, peixes, ovos, azeite de oliva, alho, cebola, ervas e especiarias.

Fontes: [Sociedade Brasileira de Diabetes](http://www2.datasus.gov.br/SIAB/index.php?area=02), [Ministério da Saúde](https://www.kaggle.com/datasets/datahackers/state-of-data-2021), [Portal Data Science](https://portaldatascience.com/kaggle/)""")

# Carregando a imagem da logo
logo = Image.open("imagens/IMG_20231110_220951.png")
# Exibindo a logo na sidebar
st.sidebar.image(logo, use_column_width=True)
# Criando um sidebar para inserir as informações do usuário
st.sidebar.header("Insira as informações do paciente:")
paciente_nome = st.sidebar.text_input("Nome do Paciente")
# Criando um seletor para o sexo
sex = st.sidebar.selectbox("Sexo", ("Feminino", "Masculino"))
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
st.divider()
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
def generate_pdf_report(paciente_nome, prediction, decision_tree_fig):
    c = canvas.Canvas(f"Diabetes_Report_{paciente_nome}.pdf", pagesize=letter)

    img_path = f"Diabetes_Tree_{paciente_nome}.png"
    decision_tree_fig.savefig(img_path, format='png')
    
    c.setFont("Helvetica", 16)
    c.drawString(100, 700, "Relatório de Previsão de Diabetes")

    c.drawImage(img_path, 50, 260, width=500, height=350)
    c.setFont("Helvetica", 12)
    c.drawString(100, 210, f"Nome do Paciente: {paciente_nome}")
    c.drawString(100, 190, f"Resultado da Previsão: {'Diabetes' if prediction == 1 else 'Sem Diabetes'}")
    c.drawString(100, 170, f"Idade: {age}")
    c.drawString(100, 150, f"Sexo: {sex}")

    c.showPage()
    c.save()

    with open(f"Diabetes_Report_{paciente_nome}.pdf", "rb") as pdf_file:
        b64_pdf = base64.b64encode(pdf_file.read()).decode()

    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Diabetes_Report_{paciente_nome}.pdf">Baixar Relatório em PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success(f"Relatório em PDF gerado com sucesso para {paciente_nome}")

#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)   
# Verificando se o botão foi clicado
if button:
    if uploaded_file is not None:
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

        # Calculando outras métricas de avaliação
        # Calculando precisão, recall e f1-score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calculando matriz de confusão e relatório de classificação
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # Mostrando as métricas na interface do Streamlit
        st.write(f"Precisão: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")
        st.write("Matriz de Confusão:")
        st.write(conf_matrix)
        st.write("Relatório de Classificação:")
        st.write(class_report)




        # Calculando a acurácia das previsões
        acc = accuracy_score(y_test, y_pred)
        st.write(f"A acurácia do modelo é {acc:.2f}")

        # Exibindo a árvore de decisão
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(clf, feature_names=X.columns.tolist(), class_names=["No", "Yes"], filled=True, rounded=True, ax=ax)
        st.pyplot(fig)

        # Fazendo a previsão para o usuário
        user_pred = clf.predict(user_data.drop("Outcome", axis=1))
        if user_pred[0] == 0:
            st.markdown("<h3 style='color: green;'>Parabéns! Paciente sem diabetes.</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: red;'>⚠️Atenção! Paciente com suspeita de diabetes.</h3>", unsafe_allow_html=True)
        #if user_pred[0] == 0:
            #st.success("Parabéns! Você não tem diabetes.")
        #else:
            #st.error("Atenção! Você tem diabetes.")
        if paciente_nome:
            generate_pdf_report(paciente_nome, user_pred[0], fig)
    else:
        st.warning("Por favor, carregue o arquivo CSV antes de fazer uma previsão.")

# Inserindo um aviso informando que é um modelo de teste
st.warning("Atenção: este aplicativo é um modelo de teste e não substitui um diagnóstico médico profissional. Consulte um médico se você tiver sintomas ou suspeita de diabetes.")
st.divider()
st.markdown("Developed by: Mauro Alves")
    
st.markdown("""
        <a href="https://github.com/Maurynn" target="_blank" style="margin-right: 15px; text-decoration: none">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="Github logo" width="25" height="25">
        </a>
        <a href="https://linkedin.com/in/maurosp" target="_blank" style="margin-right: 15px; text-decoration: none">
        <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" alt="LinkedIn logo" width="25" height="25">
        </a>
        <a href="https://instagram.com/maurinn?igshid=ZDc4ODBmNjlmNQ==" target="_blank" style="margin-right: 15px; text-decoration: none">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" alt="Instagram logo" width="25" height="25">
        </a>
        <a href="https://wa.me/5511952483074" target="_blank" style="margin-right: 15px; text-decoration: none">
        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" alt="WhatsApp logo" width="25" height="25">
        </a>
    """, unsafe_allow_html=True)
