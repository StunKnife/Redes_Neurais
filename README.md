# Introdução a Redes_Neurais

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

