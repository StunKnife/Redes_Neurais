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

 Em que $\phi_j$, $j = 0, \dots , M - 1$, são funções que dependem das funções de ativação adotadas, $\varphi(\cdot)$ é a função identidade no caso de regressão, uma função não linear no caso de classificação, e $w_j$ são pesos a serem determinados. Com essa formulação, os seguintes passos são usualmente utilizados na análise de redes neurais:

1. Considere as ativações
 $$a_j = \sum_{i=0}^{p} w_{ji}^{(1)}x_i, \quad j =1,\dots,M$$

em que, para $j = 1, \dots, M$, incluímos os vieses $w_{j0}^{(1)}$ nos vetores de pesos  $w_{j}^{(1)} = (w_{j0}^{(1)}, w_{j1}^{(1)},\dots ,w_{jp}^{(1)})^T $, fazendo $x_0 = 1$. O índice $(1)$ no expoente de $w_j$ indica a primeira camada da rede neural.

2. Cada ativação $a_j$ é transformada por meio de uma função de ativação $h(\cdot)$, resultando em:
$$y_j=h(a_j)$$
Dizemos que os $y_j$ são as unidades escondidas.
3. Considere as ativações de saída
   $$a_k = \sum_{j=0}^{M} w_{kj}^{(2)}y_j, \quad k=1,\dots,K$$
em que, novamente incluímos os vieses $w_{k0}^{(2)}$ no vetor $\textbf{w}$.
4. Finalmente, essas ativações são transformadas por meio de uma nova função de ativação $(b)$, resultando nas saídas $Z_k$ da rede neural. Para problemas de regressão, $Z_k = a_k$ e para problemas de classificação,
   $$Z_k = b(a_k)$$
com $b(a)$, em geral, correspondendo à função logística indicada acima.


Combinando os passos indicados, obtemos
$$f(x,w)=b \bigg[\sum_{j=0}^{M} w_{kj}^{(2)}h(\sum_{i=0}^{p} w_{ji}^{(1)}x_i)\bigg] $$
O procedimento utilizado é chamado  proalimentação (forward propagation) da informação.

A nomenclatura empregada em redes com essa estrutura pode diferir entre autores e pacotes computacionais. Pode ser chamada de rede neural com 3 camadas ou de rede neural com uma camada escondida. Ambas as descrições se referem à mesma arquitetura de rede neural.

# O algoritmo retropropagação (backpropagation)

O ajuste de modelos de redes neurais é baseado na minimização de uma função de perda relativamente aos pesos. No caso de regressão, essa função de perda usualmente é a soma dos quadrados dos resíduos (sum of squared errors - SSE) e no caso de classificação, em geral, usa-se a entropia. Nos dois casos, o ajuste de uma rede neural baseia-se em um algoritmo chamado de retropropagação (backpropagation). Nesse algoritmo, atualiza-se o gradiente da função de perda de maneira que os pesos sejam modificados na direção oposta àquela indicada pelo sinal do gradiente até que um mínimo seja atingido.

 - Se a derivada da função de perda for negativa, o peso é aumentado; em caso contrário, se a derivada da função de perda for positiva, o peso sofre um decréscimo.

Comumente usa-se o método do decréscimo do gradiente (gradient descent) ou suas variações, como decréscimo estocástico do gradiente (stochastic gradient descent).

## Implementação do algoritmo

Para a implementação do algoritmo, é necessário escolher valores iniciais e de regularização (usando uma função penalizadora), porque o algoritmo de otimização não é convexo e é instável. Em geral, o problema de otimização da rede neural pode ser posto na forma:
$$\hat{\textbf{w}} = \argmin_w [\tilde{Q}_n(\textbf{w}) = \argmin_w [\lambda_1Q_n(\textbf{w}) + \lambda_2Q^*(\textbf{w})]],$$
em que $\lambda_1,\lambda_2>0$,

$$Q_n(\textbf{w}) =  \sum_{i=1}^{n} (y_i - f(\textbf{x}_i, \textbf{w}))^2,$$

e $Q^*(\textbf{w})$ denota um termo de regularização, que pode ser escolhido entre: lasso, ridge ou elastic net.


Podemos idealizar uma rede neural como uma função não linear paramétrica (determinística) de uma entrada $x$, tendo $z$ como saída. Consideremos os vetores do conjunto de treinamento, $x_i$, os vetores alvos (saídas) $z_i, i = 1, \dots , n$ e a soma dos quadrados dos erros que queremos minimizar, ligeiramente modificada como

$$Q_n(\textbf{w}) = \dfrac{1}{2}\sum_{i=1}^{n} ||z_i - f(\textbf{x}_i, \textbf{w})||^2.$$

### Problema de regressão

Tratemos primeiramente o problema de regressão, considerando a rede neural com um erro aleatório $e_i$ acrescentado antes da saída, de modo a associar ao algoritmo um modelo probabilístico. Por simplicidade, consideremos a saída $z = (z_1, \dots , z_n)^T$ e os erros com distribuição Normal, com média zero e variância $\sigma^2$, de modo que
$$z \sim N_n[f(\textbf{x}_i, \textbf{w}), \sigma^2 I]$$

Admitamos ainda que a função de ativação de saída é a função identidade. Organizando os vetores de entrada na matriz $x = [x_1, \dots, x_n]$, a verossimilhança pode ser escrita como
$$L(\textbf{z}|\textbf{X}, \textbf{w}, \sigma^2) = \prod_{i=1}^{n} \phi(z_i|\textbf{x}_i, \textbf{w}, \sigma^2)$$

Como resultado da minimização, obtemos o estimador de máxima verossimilhança $\hat{\textbf{w}}_{MV}$ dos pesos $w$ e consequentemente, o estimador de máxima verossimilhança de $\sigma^2$. Como $Q_n(\textbf{w})$ é uma função não linear não convexa, podemos obter máximos não locais da verossimilhança.

### Problema de classificação binária

No caso de classificação binária, para a qual, por exemplo, $z = +1$ indica a classe $C_1$ e $z = 0$ indica a classe $C_2$, consideremos uma rede neural com saída única $z$ com função de ativação logística,

$$b(a) = 1 / (1 + e^{-a}),$$

de modo que $0 \le f(\textbf{x}$, $\textbf{w}) \le 1$. Podemos interpretar $f(\textbf{x}, \textbf{w})$ como $P(z = 1|x)$ e $1 − f(\textbf{x}, \textbf{w})$ como $P(z = 0|x)$. A distribuição de $z$, dado $x$, é

$$f(z|\textbf{x},\textbf{w}) = f(\textbf{x}, \textbf{w})^z [1 − f(\textbf{x}, \textbf{w})]^{1-z}$$

Considerando as observações de treinamento independentes e identicamente distribuídas, a função de perda usual é a entropia cruzada

$$Q_n(\textbf{w}) = - \sum_{i=1}^{n} [\nu_i  \log(\nu_i) - (1 - \nu_i)  \log(1 - \nu_i)],$$

onde $\nu_i = f(\textbf{x}_i$, $\textbf{w})$.

Para otimizar os pesos $w$, ou seja, encontrar o valor que minimiza $Q_n(\textbf{w})$, usualmente, precisamos obter o gradiente de $Q_n$, denotado $\nabla Q_n$ e que aponta para a maior taxa de aumento de $Q_n$. Supondo que $Q_n$ seja uma função contínua e suave de $w$, o valor mínimo ocorre no ponto em que o gradiente se anula. Em geral, procedimentos numéricos são usados com essa finalidade e há uma extensa literatura sobre o assunto. As técnicas que usam o gradiente começam fixando um valor inicial $\textbf{w}^{(0)}$ para os pesos que são iterativamente atualizados por meio de

$$\textbf{w}^{(r+1)} = \textbf{w}^{(r)} − \lambda\nabla Q_n(\textbf{w}^{(r)}),$$

em que $\lambda$ é chamada de taxa de aprendizado. Esse método, chamado de método de decréscimo do gradiente, usa todo o conjunto de treinamento, mas não é muito eficiente. Na prática, usa-se o algoritmo de retropropagação para calcular o gradiente em uma rede neural.

O problema de classificação é bem mais complexo quando a variável resposta tem $K > 2$ categorias. As soluções mais simples consistem em ajustar $M$ redes neurais com uma das seguintes estratégias:


1. OAA: Uma contra todas (One Against All), em que cada uma das $M = K$ redes neurais envolve a categoria $i$ contra as demais.

2. OAO: Uma contra outra (One Against One), em que cada uma das $M = K(K + 1)/2$ redes envolve a categoria $i$ contra a categoria $j$, $i, j = 1, \dots, K, i \ne j$.

3. PAQ: $P$ categorias contra $Q$ categorias ($P$ Against $Q$), em que as categorias de resposta são agrupadas em duas categorias.

# Aprendizado profundo  (Deep learning)

No caso de redes neurais com várias camadas intermediárias, obtém-se o que é chamado de aprendizado profundo (deep learning), no qual o sistema computacional "aprende" a partir dos dados (e exemplos) em termos de uma hierarquia de conceitos, cada um definido por meio de sua relação com conceitos mais simples, ou ainda, "aprende" conceitos complicados a partir de conceitos simples. Um exemplo de modelo de aprendizado profundo é o perceptron multicamadas (multilayer perceptron), que essencialmente é uma aplicação que transforma os dados de entrada em valores de saída por meio da composição de funções simples.

Para analisar redes neurais com várias camadas, é possível usar o pacote Keras do R, que por sua vez usa o pacote TensorFlow com capacidades CPU e GPU. Modelos de deep learning que podem ser analisados incluem redes neurais recorrentes (recurrent neural networks - RNN), redes Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), Multilayer Perceptron, etc. As redes LSTM são apropriadas para captar dependências de longo prazo e implementar previsão de séries temporais.


Os neurônios artificiais consistem principalmente em duas partes: soma e ativação. A soma envolve a adição de todos os sinais de entrada, enquanto a ativação determina se o neurônio irá disparar com base em um valor limite. Vamos considerar um cenário com duas entradas binárias ($X1$, $X2$) e pesos para suas respectivas conexões ($W1, W2$). Esses pesos são análogos aos coeficientes das variáveis de entrada no aprendizado de máquina tradicional e significam a importância dos recursos de entrada específicos no modelo.

A função de soma calcula a soma total dos sinais de entrada e a função de ativação usa essa soma para produzir uma saída. A ativação atua como uma função de tomada de decisão e a escolha da função de ativação determina a saída. Existem vários tipos de funções de ativação que podem ser empregadas em uma camada de rede neural. Como vimos, em cenários do mundo real, a relação dificilmente é simples e linear. Portanto, devemos introduzir uma camada adicional de neurônios entre a camada de entrada e a camada de saída, a fim de aumentar a capacidade da rede de aprender diferentes tipos de relações não lineares. Essa camada adicional de neurônios é conhecida como camada oculta

