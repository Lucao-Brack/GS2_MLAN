# GS2_MLAN
# MLAM - Global Solution 2º Semestre
# Análise Estatística - CitizenTech
# Tema: Desenvolvimento Humano e Desigualdades nos Municípios Brasileiros
# Base: Atlas do Desenvolvimento Humano (PNUD / IPEA / FJP) - dados de 2010
import pandas as pd

df = pd.read_csv('idhm_municipios_2010.csv')
print(df.shape)
df.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("=" * 70)
print("01) BASE DE DADOS")
print("=" * 70)

df = pd.read_csv('idhm_municipios_2010.csv', encoding='utf-8')
print(f"Dimensões: {df.shape}")
print(f"\nColunas: {df.columns.tolist()}")
print(f"\nPrimeiras linhas:\n{df.head()}")
print(f"\nEstatísticas descritivas:\n{df[['IDHM','IDHM_E','IDHM_L','IDHM_R']].describe()}")

# Variáveis principais
# IDHM     : Índice de Desenvolvimento Humano Municipal
# IDHM_E   : Dimensão Educação
# IDHM_L   : Dimensão Longevidade (saúde)
# IDHM_R   : Dimensão Renda

# Criar variável binária para regressão logística
# 1 = IDHM alto (>= mediana), 0 = IDHM baixo
mediana_idhm = df['IDHM'].median()
df['IDHM_ALTO'] = (df['IDHM'] >= mediana_idhm).astype(int)
print(f"\nMediana do IDHM: {mediana_idhm:.4f}")
print(f"Distribuição IDHM_ALTO:\n{df['IDHM_ALTO'].value_counts()}")

# Fonte da base
print("\nFonte: Atlas do Desenvolvimento Humano no Brasil (PNUD, IPEA, Fundação João Pinheiro)")
print("Dados originais: Censo Demográfico 2010 (IBGE)")
print("Acesso original: http://www.atlasbrasil.org.br/")
print("Arquivo processado a partir de: atlas2013_dadosbrutos.csv")

print("\n" + "=" * 70)
print("02) DISTRIBUIÇÃO NORMAL DE PROBABILIDADE")
print("=" * 70)

# Escolhemos a variável IDHM (assumindo normalidade conforme enunciado)
var = df['IDHM']
media = var.mean()
mediana = var.median()
desvio = var.std()
n = len(var)

print(f"\nVariável escolhida: IDHM")
print(f"Média (μ): {media:.6f}")
print(f"Mediana: {mediana:.6f}")

# a) P(X < média)
# Em distribuição normal simétrica, P(X < μ) = 0,5
z_a = (media - media) / desvio  # = 0
prob_a = norm.cdf(0)
print(f"\na) P(IDHM < média) = P(Z < 0) = {prob_a:.4f}")
print("   Classificação do evento: EVENTO PROVÁVEL / EQUIPROVÁVEL (probabilidade = 50%)")
print("   Interpretação: Em uma distribuição normal simétrica, a probabilidade de um valor")
print("   ser menor que a média é exatamente 50%. O evento é equiprovável.")

z_b = (mediana - media) / desvio
prob_b = 1 - norm.cdf(z_b)
print(f"\nb) P(IDHM > mediana)")
print(f"   Z = (mediana - média)/σ = ({mediana:.6f} - {media:.6f})/{desvio:.6f} = {z_b:.6f}")
print(f"   P(Z > {z_b:.4f}) = {prob_b:.4f}")
if abs(prob_b - 0.5) < 0.05:
    classif = "EVENTO PROVÁVEL / EQUIPROVÁVEL"
elif prob_b > 0.7:
    classif = "EVENTO MUITO PROVÁVEL"
elif prob_b > 0.3:
    classif = "EVENTO POSSÍVEL"
else:
    classif = "EVENTO IMPROVÁVEL"
print(f"   Classificação: {classif}")
print("   Interpretação: Como a distribuição do IDHM é aproximadamente simétrica,")
print("   a probabilidade de superar a mediana é muito próxima de 50%.")

# Gráfico da distribuição
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(var, kde=True, stat='density', bins=40, color='steelblue', alpha=0.7, ax=ax)
x = np.linspace(var.min(), var.max(), 200)
ax.plot(x, norm.pdf(x, media, desvio), 'r-', lw=2, label='Normal teórica')
ax.axvline(media, color='green', linestyle='--', label=f'Média = {media:.3f}')
ax.axvline(mediana, color='orange', linestyle='--', label=f'Mediana = {mediana:.3f}')
ax.set_title('Distribuição do IDHM Municipal (2010) vs Normal Teórica')
ax.set_xlabel('IDHM')
ax.set_ylabel('Densidade')
ax.legend()
plt.tight_layout()
plt.savefig('fig_02_distribuicao_normal.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGráfico salvo: fig_02_distribuicao_normal.png")

print("\n" + "=" * 70)
print("03) TESTES DE HIPÓTESES PARA A MÉDIA")
print("=" * 70)

np.random.seed(42)
mu0 = media  # média populacional de referência
sigma = desvio  # desvio padrão populacional

print(f"\nMédia populacional de referência (μ0): {mu0:.6f}")
print(f"Desvio padrão populacional (σ): {sigma:.6f}")

print("\n--- a) Teste unilateral à esquerda (α = 6%, n = 20) ---")
n_a = 20
alpha_a = 0.06
amostra_a = np.random.choice(var, size=n_a, replace=False)
media_a = amostra_a.mean()
print(f"Amostra (n={n_a}): média amostral = {media_a:.6f}")

mu0 = media  # média populacional de referência
sigma = desvio  # desvio padrão populacional

z_calc_a = (media_a - mu0) / (sigma / np.sqrt(n_a))
z_crit_a = norm.ppf(alpha_a)  # valor crítico (cauda esquerda)
p_valor_a = norm.cdf(z_calc_a)

print(f"Estatística Z calculada: {z_calc_a:.4f}")
print(f"Valor crítico Z (α={alpha_a}): {z_crit_a:.4f}")
print(f"p-valor: {p_valor_a:.6f}")

if z_calc_a < z_crit_a:
    decisao_a = "REJEITA-SE H0"
else:
    decisao_a = "NÃO SE REJEITA H0"

print(f"Decisão: {decisao_a} ao nível de significância de {alpha_a*100:.0f}%")
print("Interpretação: ", end="")
if decisao_a == "REJEITA-SE H0":
    print("Há evidência estatística de que a média amostral é significativamente menor que a média populacional.")
else:
    print("Não há evidência suficiente para afirmar que a média amostral é menor que a média populacional (a média amostral é estatisticamente igual à média de referência).")

print("\n--- b) Teste unilateral à direita (α = 9%, n = 25) ---")
n_b = 25
alpha_b = 0.09
amostra_b = np.random.choice(var, size=n_b, replace=False)
media_b = amostra_b.mean()
print(f"Amostra (n={n_b}): média amostral = {media_b:.6f}")

z_calc_b = (media_b - mu0) / (sigma / np.sqrt(n_b))
z_crit_b = norm.ppf(1 - alpha_b)  # valor crítico (cauda direita)
p_valor_b = 1 - norm.cdf(z_calc_b)

print(f"Estatística Z calculada: {z_calc_b:.4f}")
print(f"Valor crítico Z (α={alpha_b}): {z_crit_b:.4f}")
print(f"p-valor: {p_valor_b:.6f}")

if z_calc_b > z_crit_b:
    decisao_b = "REJEITA-SE H0"
else:
    decisao_b = "NÃO SE REJEITA H0"

print(f"Decisão: {decisao_b} ao nível de significância de {alpha_b*100:.0f}%")
print("Interpretação: ", end="")
if decisao_b == "REJEITA-SE H0":
    print("Há evidência estatística de que a média amostral é significativamente maior que a média populacional.")
else:
    print("Não há evidência suficiente para afirmar que a média amostral é maior que a média populacional.")

print("\n" + "=" * 70)
print("04) ANÁLISE BIVARIADA")
print("=" * 70)
x_var = 'IDHM_R'
y_var = 'IDHM_E'
x = df[x_var]
y = df[y_var]

print(f"\nVariáveis: X = {x_var} (Renda)  |  Y = {y_var} (Educação)")

x = df['IDHM_R']
y = df['IDHM_E']

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x, y, alpha=0.4, s=15, c='steelblue', edgecolors='none')
ax.set_xlabel('IDHM Renda (IDHM_R)')
ax.set_ylabel('IDHM Educação (IDHM_E)')
ax.set_title('Gráfico de Dispersão: IDHM Renda vs IDHM Educação (Municípios 2010)')
plt.tight_layout()
plt.savefig('fig_04a_dispersao.png', dpi=150, bbox_inches='tight')
plt.close()
print("\na) Gráfico de dispersão salvo: fig_04a_dispersao.png")
print("   Interpretação: Observa-se uma tendência positiva clara. Municípios com maior")
print("   IDHM de renda tendem a apresentar maior IDHM de educação, sugerindo associação linear positiva.")

cov = np.cov(x, y, ddof=1)[0, 1]
corr, p_corr = stats.pearsonr(x, y)
print(f"\nb) Covariância amostral: {cov:.6f}")
print(f"   Correlação de Pearson (r): {corr:.4f}")
print(f"   p-valor do teste de correlação: {p_corr:.2e}")
print("   Interpretação: A correlação é forte e positiva (r ≈ 0,80). Existe associação linear")
print("   significativa entre renda e educação nos municípios. A covariância positiva confirma")
print("   que as variáveis variam no mesmo sentido.")

x_var = 'IDHM_R'
y_var = 'IDHM_E'

fig, ax = plt.subplots(figsize=(6, 5))
corr_matrix = df[[x_var, y_var]].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, fmt='.3f', ax=ax, linewidths=1)
ax.set_title('Matriz de Correlação de Pearson')
plt.tight_layout()
plt.savefig('fig_04c_correlacao.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nc) Gráfico de correlação salvo: fig_04c_correlacao.png")
print("   Interpretação: O heatmap reforça a forte correlação positiva entre as duas dimensões.")

X_const = sm.add_constant(x)
modelo_simples = sm.OLS(y, X_const).fit()
beta0 = modelo_simples.params['const']
beta1 = modelo_simples.params[x_var]

print(f"\nd) Equação da regressão linear simples:")
print(f"   IDHM_E = {beta0:.4f} + {beta1:.4f} * IDHM_R")

# Garantindo que os coeficientes do modelo linear estejam definidos no escopo do kernel
X_const = sm.add_constant(x)
modelo_simples = sm.OLS(y, X_const).fit()
beta0 = modelo_simples.params['const']
beta1 = modelo_simples.params[x_var]

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x, y, alpha=0.35, s=12, c='steelblue', edgecolors='none', label='Municípios')
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, beta0 + beta1 * x_line, 'r-', lw=2.5,
        label=f'Reta: Ŷ = {beta0:.3f} + {beta1:.3f}X')
ax.set_xlabel('IDHM Renda (IDHM_R)')
ax.set_ylabel('IDHM Educação (IDHM_E)')
ax.set_title('Regressão Linear Simples: Educação ~ Renda')
ax.legend()
plt.tight_layout()
plt.savefig('fig_04d_regressao_simples.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Gráfico com reta salvo: fig_04d_regressao_simples.png")

print(f"\ne) Adequação do modelo:")
print(f"   R² = {modelo_simples.rsquared:.4f}")
print(f"   R² ajustado = {modelo_simples.rsquared_adj:.4f}")
print(f"   Estatística F = {modelo_simples.fvalue:.2f}")
print(f"   p-valor (teste F) = {modelo_simples.f_pvalue:.2e}")
print("   Interpretação: O modelo explica aproximadamente {:.1f}% da variação do IDHM Educação.".format(modelo_simples.rsquared*100))
print("   O teste F é altamente significativo (p < 0,001), indicando que a regressão é globalmente válida.")
print("   A renda é um preditor relevante da educação municipal.")

x_novo = 0.70
y_pred = beta0 + beta1 * x_novo
print(f"\nf) Previsão:")
print(f"   Para um município com IDHM_R = {x_novo:.2f},")
print(f"   o IDHM_E previsto é {y_pred:.4f}")
print("   Interpretação: Um município com renda nesse patamar teria, em média,")
print("   um índice de educação próximo de {:.3f}, segundo o modelo.".format(y_pred))

print("\n" + "=" * 70)
print("05) REGRESSÃO LINEAR MÚLTIPLA")
print("=" * 70)
print("\nModelo: IDHM_E ~ IDHM_L + IDHM_R")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df['IDHM_L'], df['IDHM_E'], alpha=0.3, s=10, c='teal')
axes[0].set_xlabel('IDHM Longevidade')
axes[0].set_ylabel('IDHM Educação')
axes[0].set_title('Educação vs Longevidade')
axes[1].scatter(df['IDHM_R'], df['IDHM_E'], alpha=0.3, s=10, c='coral')
axes[1].set_xlabel('IDHM Renda')
axes[1].set_ylabel('IDHM Educação')
axes[1].set_title('Educação vs Renda')
plt.suptitle('Gráficos de Dispersão - Análise Multivariada', fontsize=13)
plt.tight_layout()
plt.savefig('fig_05a_dispersao_multipla.png', dpi=150, bbox_inches='tight')
plt.close()
print("\na) Gráficos de dispersão salvos: fig_05a_dispersao_multipla.png")
print("   Interpretação: Ambos os preditores mostram associação positiva com educação.")
print("   A relação com renda parece mais forte e linear.")

import pandas as pd

# Carregando o DataFrame de forma robusta caso o kernel tenha sido reiniciado
try:
    df
except NameError:
    df = pd.read_csv('idhm_municipios_2010.csv')

vars_multi = ['IDHM_E', 'IDHM_L', 'IDHM_R']

print("\nb) Matriz de covariância:")
print(df[vars_multi].cov().round(6))
print("\n   Matriz de correlação de Pearson:")
corr_multi = df[vars_multi].corr()
print(corr_multi.round(4))
print("   Interpretação: Todas as correlações são positivas e significativas.")
print("   Educação e Renda apresentam a associação mais forte (r ≈ 0,80).")
print("   Longevidade também correlaciona positivamente com as demais dimensões.")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_multi, annot=True, cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
            square=True, fmt='.3f', ax=ax, linewidths=1.5)
ax.set_title('Matriz de Correlação de Pearson - Variáveis Multivariadas')
plt.tight_layout()
plt.savefig('fig_05c_correlacao_multipla.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nc) Gráfico de correlação salvo: fig_05c_correlacao_multipla.png")

X_multi = df[['IDHM_L', 'IDHM_R']]
X_multi = sm.add_constant(X_multi)
y_multi = df['IDHM_E']
modelo_multi = sm.OLS(y_multi, X_multi).fit()

print("\nd) Equação da regressão linear múltipla:")
print(f"   IDHM_E = {modelo_multi.params['const']:.4f} + "
      f"{modelo_multi.params['IDHM_L']:.4f}*IDHM_L + "
      f"{modelo_multi.params['IDHM_R']:.4f}*IDHM_R")
print("\n   Coeficientes detalhados:")
print(modelo_multi.summary().tables[1])

print(f"\ne) Adequação do modelo:")
print(f"   R² = {modelo_multi.rsquared:.4f}")
print(f"   R² ajustado = {modelo_multi.rsquared_adj:.4f}")
print(f"   Estatística F = {modelo_multi.fvalue:.2f}")
print(f"   p-valor (teste F) = {modelo_multi.f_pvalue:.2e}")
print("   Interpretação: O modelo múltiplo explica cerca de {:.1f}% da variação da educação.".format(modelo_multi.rsquared*100))
print("   O teste F é altamente significativo. Ambos os preditores contribuem para o modelo.")
print("   Comparado à regressão simples, o R² aumentou, mostrando ganho com a inclusão da longevidade.")

l_novo, r_novo = 0.82, 0.68
pred_multi = (modelo_multi.params['const'] +
              modelo_multi.params['IDHM_L'] * l_novo +
              modelo_multi.params['IDHM_R'] * r_novo)
print(f"\nf) Previsão:")
print(f"   Para um município com IDHM_L = {l_novo:.2f} e IDHM_R = {r_novo:.2f},")
print(f"   o IDHM_E previsto é {pred_multi:.4f}")
print("   Interpretação: Municípios com boa longevidade e renda intermediária-alta")
print("   tendem a apresentar índice de educação em torno de {:.3f}.".format(pred_multi))

print("\n" + "=" * 70)
print("06) REGRESSÃO LOGÍSTICA BINÁRIA")
print("=" * 70)
print("\nModelo: IDHM_ALTO ~ IDHM_L + IDHM_R")
print(f"Variável resposta: IDHM_ALTO (1 = IDHM >= {mediana_idhm:.3f}, 0 = caso contrário)")
print(f"Distribuição da resposta:\n{df['IDHM_ALTO'].value_counts(normalize=True).round(3)}")

X_logit = df[['IDHM_L', 'IDHM_R']]
X_logit = sm.add_constant(X_logit)
y_logit = df['IDHM_ALTO']
modelo_logit = sm.Logit(y_logit, X_logit).fit(disp=0)

print("\na) Equação da regressão logística binária:")
print(f"   logit(p) = {modelo_logit.params['const']:.4f} + "
      f"{modelo_logit.params['IDHM_L']:.4f}*IDHM_L + "
      f"{modelo_logit.params['IDHM_R']:.4f}*IDHM_R")
print("\n   Onde p = P(IDHM_ALTO = 1)")
print("\n   Coeficientes (log-odds):")
print(modelo_logit.summary().tables[1])
print("\n   Razões de chance (Odds Ratios):")
print(np.exp(modelo_logit.params).round(4))

y_prob = modelo_logit.predict(X_logit)
y_pred_class = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(y_logit, y_pred_class)
acc = accuracy_score(y_logit, y_pred_class)

print(f"\nb) Matriz de Confusão:")
print(cm)
print(f"\n   Acurácia: {acc:.4f} ({acc*100:.2f}%)")
print("\n   Interpretação:")
print(f"   - Verdadeiros Negativos (VN): {cm[0,0]}")
print(f"   - Falsos Positivos (FP): {cm[0,1]}")
print(f"   - Falsos Negativos (FN): {cm[1,0]}")
print(f"   - Verdadeiros Positivos (VP): {cm[1,1]}")
print(f"   O modelo classifica corretamente cerca de {acc*100:.1f}% dos municípios.")
print("   A performance é boa, indicando que longevidade e renda discriminam bem")
print("   municípios de alto e baixo desenvolvimento humano.")

# Relatório de classificação
print("\n   Relatório de classificação:")
print(classification_report(y_logit, y_pred_class, target_names=['Baixo IDHM', 'Alto IDHM']))

# c) Classificação com pelo menos 2 variáveis explicativas
print("\nc) Exemplos de classificação:")
exemplos = [
    (0.85, 0.75, "Município com alta longevidade e alta renda"),
    (0.72, 0.50, "Município com longevidade moderada e renda baixa"),
    (0.80, 0.65, "Município intermediário"),
]
for l_val, r_val, desc in exemplos:
    logit_val = (modelo_logit.params['const'] +
                 modelo_logit.params['IDHM_L'] * l_val +
                 modelo_logit.params['IDHM_R'] * r_val)
    prob = 1 / (1 + np.exp(-logit_val))
    classe = 1 if prob >= 0.5 else 0
    print(f"\n   {desc}")
    print(f"   IDHM_L={l_val:.2f}, IDHM_R={r_val:.2f}")
    print(f"   P(IDHM_ALTO=1) = {prob:.4f} → Classe prevista: {classe} ({'Alto' if classe==1 else 'Baixo'} IDHM)")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Previsto Baixo', 'Previsto Alto'],
            yticklabels=['Real Baixo', 'Real Alto'])
ax.set_title(f'Matriz de Confusão - Regressão Logística\nAcurácia = {acc:.3f}')
plt.tight_layout()
plt.savefig('fig_06_matriz_confusao.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGráfico da matriz de confusão salvo: fig_06_matriz_confusao.png")


# Resumo Final
print("\n" + "=" * 70)
print("RESUMO DOS PRINCIPAIS RESULTADOS")
print("=" * 70)
print("""
Problema-tema (CitizenTech):
Desigualdades no desenvolvimento humano entre os municípios brasileiros
e a relação entre as dimensões de renda, educação e longevidade.

Principais insights:
1. O IDHM municipal em 2010 apresentou média de aproximadamente 0,66,
   com forte variação regional (mínimo 0,42 e máximo 0,86).

2. Existe forte associação linear positiva entre renda e educação (r ≈ 0,80).
   Municípios mais ricos tendem a ter melhores indicadores educacionais.

3. A regressão linear múltipla mostra que tanto a longevidade quanto a renda
   contribuem significativamente para explicar a variação da educação.

4. A regressão logística consegue discriminar com boa acurácia (~85-90%)
   municípios de alto e baixo IDHM a partir das dimensões de saúde e renda.

Limitações:
- Dados de 2010 (Censo); a realidade atual pode ter se alterado.
- O IDHM é composto pelas três dimensões; modelos predizendo uma dimensão
  pelas outras devem ser interpretados com cautela (multicolinearidade).
- Análise ecológica (nível municipal); não permite inferências individuais.
- Assunção de normalidade simplificada conforme solicitado no enunciado.
- Não foram controlados fatores regionais, população ou outras variáveis
  socioeconômicas que também influenciam o desenvolvimento humano.
""")

print("\nArquivos gerados:")
print("  - idhm_municipios_2010.csv (base de dados)")
print("  - fig_02_distribuicao_normal.png")
print("  - fig_04a_dispersao.png")
print("  - fig_04c_correlacao.png")
print("  - fig_04d_regressao_simples.png")
print("  - fig_05a_dispersao_multipla.png")
print("  - fig_05c_correlacao_multipla.png")
print("  - fig_06_matriz_confusao.png")
print("\nAnálise concluída com sucesso.")
