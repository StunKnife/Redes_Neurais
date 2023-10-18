# Introdução a Redes Neurais

Uma rede neural é um conjunto de algoritmos construídos para identificar relações
entre as variáveis de um conjunto de dados por intermédio de um processo que
tenta imitar a maneira com que neurônios interagem no cérebro.

Cada "neurônio" em uma rede neural é uma função à qual dados são alimentados e transformados em uma resposta, como em modelos de regressão. Essa resposta é repassada para um ou mais neurônios da rede que, ao final do processo, produzem um resultado ou uma recomendação. Por exemplo, 
  - Para um conjunto de clientes de um banco que solicitam empréstimos (as entradas [inputs]) (saldo médio, idade, nível educacional etc.) são alimentados à rede neural que gera saídas (outputs) do tipo: "negar o empréstimo", "conceder o empréstimo" ou "solicitar mais informações".

## Arquitetura de uma rede neural
Redes neurais são caracterizadas por uma arquitetura que corresponde à maneira como os neurônios estão organizados em camadas. Há três classes de arquiteturas comumente utilizadas:

  1. **Rede neural com uma camada de entrada e uma de saída** (conhecida como perceptron).
  2. **Rede neural multicamadas**, também conhecida por rede do tipo proalimentada (feedforward), em que há uma ou mais camadas escondidas com as entradas de cada neurônio de uma camada obtidas de neurônios da camada precedente.
  3. **Rede neural recorrente**, em que pelo menos um neurônio conecta-se com um neurônio da camada precedente, criando um ciclo de retroalimentação (feedback).

# Perceptron

Rosenblatt (1958) introduziu o algoritmo perceptron como o primeiro modelo de aprendizado supervisionado. A ideia do algoritmo é atribuir pesos $w_i$ aos dados de entrada $x$, iterativamente, para a tomada de decisão. 

## Perceptron: Classificação Binária
No caso de classificação binária, o objetivo é classificar um dado de entrada em uma de duas classes, rotuladas aqui por $+1$ e $-1$. O modelo subjacente consiste em uma combinação linear das entradas, incorporando um viés externo. O resultado dessa combinação linear é comparado com um limiar, definido por meio de uma função de ativação, que essencialmente é uma função degrau ou sigmoide.

Se $x = (1, x_1, \dots , x_p)^T$ > representa um dado de entrada e $\textbf{w} = (−b, w_1, \dots , w_p)^T$ >, o vetor de pesos associados a cada um de seus elementos, calcula-se, inicialmente a combinação linear
	$$\nu = \sum_{i=0}^{p} w_ix_i=\textbf{w}^Tx$$

Dada uma função de ativação $f(\nu)$, o elemento cujos dados geraram o valor $\nu$ é classificado na classe +1 se $f(\nu) \ge b$ ou na classe -1 se $f(\nu) < b$, ou seja,
  1. $f(\nu) = +1$ se $\nu  \ge b$
  2. $f(\nu) = -1$ se $\nu  < b$

## Algoritmo do Perceptron

O algoritmo Perceptron consiste nos seguintes passos:

i) Inicialize todos os pesos como sendo zero ou valores aleatórios pequenos.

ii) Para cada dado $xi = (x_{i1}, \dots , x_{ip})$ do conjunto de treinamento:
a) Calcule os valores de saída por meio de $\nu$ e $f(\nu)$.
b) Atualize os pesos segundo a seguinte regra de aprendizado:
$\Delta_{wj} = \eta(alvo_i - saída_i)x_{ij},$

Em que $\eta$ é a taxa de aprendizado (um valor entre 0 e 1), alvo é o verdadeiro rótulo da classe ($y_i$), e saída é o rótulo da classe prevista. Todos os pesos são atualizados simultaneamente. Por exemplo, para duas variáveis, $x_{i1} e x_{i2}$, os pesos $w_0$, $w_1$ e $w_2$ devem ser atualizados. Nos casos em que o perceptron prevê o rótulo da classe verdadeira, $\Delta w_j = 0$ para todo $j$. Nos casos com previsão incorreta, $\Delta w_j = 2\eta x_{ij}$ ou $\Delta w_j = -2\eta x_{ij}$.

$\textbf{OBS:}$ Embora a taxa de aprendizado afete a convergência dos algoritmos mais gerais de redes neurais, pode-se demonstrar que sua escolha não muda a convergência do perceptron, e por esse motivo, utiliza-se $\eta = 1$ na sua implementação. A convergência do perceptron somente é garantida se as duas classes forem linearmente separáveis. Se não forem, podemos fixar um número máximo de iterações (conhecidas como "épocas") ou um limiar para o número máximo tolerável de classificações erradas.

# Redes com camadas escondidas

Uma das redes neurais mais simples consiste de entradas, de uma camada intermediária escondida e de saídas. Os elementos de $x = (x_1,\dots , x_p)^T$ indicam as entradas, aqueles de $z = (z_1, \dots, z_K)^T$ denotam as saídas, os de $y = (y_1, . . . , y_M)^T$ (não observáveis) constituem a camada escondida, e $\alpha_j = (\alpha_{j1}, . . . , \alpha_{jp})^T$, $j = 1, \dots, M$, $\beta_k = (\beta_{k1},\dots, \beta_{kM})^T$, $k = 1, \dots , K$, respectivamente, são os pesos.

$\textbf{OBS:}$ Para regressão, há uma única saída ($K = 1$); para classificação binária, $K = 2$, e $Z$ pode ter dois valores, +1 ou -1. Para classificação em mais do que 2 classes, $K > 2$. 

A rede neural com mais de 2 classes, $K>2$, pode ser descrita pelas equações:

$$Y_j = h(\alpha_{0j} + \alpha_j^T \textbf{X}), j = 1, \dots, M, $$
$$Z_k = g(\beta_{0k} + \beta_k^T \textbf{Y}), k = 1, \dots , K. $$
Aqui, o segundo $\alpha$ e $\beta$ são os vetores.

As funções $h$ e $g$ são as funções de ativação, e as mais comumente empregadas neste contexto são:

$\textbf{Função logística (ou sigmoide):}$ $f(x) = (1 + e^{-x})^{-1}$

$\textbf{Função tangente hiperbólica:}$ $f(x) = (e^x - e^{-x})) / (e^x + e^{-x})$

$\textbf{Função ReLU (rectified linear unit):}$ $f(x) = max(0, x)$

$\textbf{Função ReLU com vazamento (leaky ReLU):}$ $f(x) = x$ se $x>0$, $f(x) = 0,01x$ se $x<0$

A função ReLU com vazamento é bastante utilizada, pois seu gradiente pode ser facilmente calculado e permite uma otimização mais rápida do que aquela associada à função sigmoide. Entretanto, ela não é derivável na origem, e no algoritmo de retroalimentação, por exemplo, necessitamos de derivabilidade.  Os pesos $\alpha_{0j}$ e $\beta_{0k}$ têm o mesmo papel de $b$ no perceptron e representam vieses.

Podemos considerar a saída da rede neural expressa na forma
	$$f(\textbf{x},\textbf{w}) = \varphi(\sum_{j=0}^{M-1} w_j\phi_j(x))$$
