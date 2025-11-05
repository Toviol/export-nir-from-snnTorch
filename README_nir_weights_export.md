# Exportação de Pesos do NIR para C Header

Este documento explica como usar a funcionalidade de exportação de pesos de arquivos NIR para arquivos `.h` compatíveis com C/C++.

## Visão Geral

A funcionalidade foi adicionada ao arquivo `nir_to_c.py` seguindo o mesmo padrão utilizado no `export_weights.py`, permitindo extrair pesos de arquivos `.nir` e gerar arquivos `.h` organizados.

## Como Usar

### Método 1: Função Standalone (Recomendado)

```python
from nir_to_c import export_nir_weights_to_c_header

# Exportação simples
export_nir_weights_to_c_header(
    nir_file="csnn_mnist.nir",
    header_path="meus_pesos.h",
    ctype="float",
    verbose=True
)
```

### Método 2: Usando a Classe NIRToCppParser

```python
from nir_to_c import NIRToCppParser

parser = NIRToCppParser("csnn_mnist.nir")
parser.export_weights_to_c_header(
    header_path="meus_pesos.h",
    ctype="float",
    verbose=True
)
```

### Método 3: Executar o Script Completo

```bash
cd /home/ana/tobias/snnProject
python export_nir_from_snnTorch/nir_to_c.py
```

## Parâmetros Disponíveis

- `nir_file`: Caminho para o arquivo .nir de entrada
- `header_path`: Caminho para o arquivo .h de saída (padrão: "nir_weights.h")
- `ctype`: Tipo C para os pesos - "float", "double", "weight_t" (padrão: "float")
- `emit_typedef_if_builtin`: Se deve emitir typedef para tipos builtin (padrão: True)
- `line_wrap`: Número de valores por linha nos arrays (padrão: 10)
- `float_fmt`: Formato para números float (padrão: ".8f")
- `verbose`: Se deve imprimir informações detalhadas (padrão: True)

## Tipos de Dados Exportados

O script extrai e exporta os seguintes tipos de dados do arquivo NIR:

1. **Pesos de camadas**: `weight` (Conv2d, Affine/Linear)
2. **Bias de camadas**: `bias` (Conv2d, Affine/Linear)
3. **Parâmetros de neurônios LIF**:
   - `tau`: Constante de tempo
   - `v_threshold`: Limiar de tensão
   - `v_leak`: Tensão de vazamento

## Estrutura do Arquivo de Saída

O arquivo `.h` gerado contém:

1. **Guards de inclusão**: Evita inclusão múltipla
2. **Metadados de dimensões**: `#define` para cada dimensão dos tensores
3. **Arrays organizados**: 
   - Tensores 1D: arrays simples
   - Tensores 2D: matrizes bidimensionais
   - Tensores 4D: arrays 4D (para Conv2d)
4. **Estatísticas**: Min/max de cada tensor (no verbose)

## Exemplo de Saída

```c
#ifndef NIR_WEIGHTS_H
#define NIR_WEIGHTS_H

// Número de tensores exportados do NIR
#define NIR_NUM_TENSORS 15

#define LAYER_0_WEIGHT_DIM0 12
#define LAYER_0_WEIGHT_DIM1 1
#define LAYER_0_WEIGHT_DIM2 5
#define LAYER_0_WEIGHT_DIM3 5
#define LAYER_0_WEIGHT_NDIMS 4

float layer_0_weight[12][1][5][5] = {
  {
    {
      {-0.92344916,-0.68668306,-0.51498026,-0.60281235,-0.54794061},
      // ... mais valores ...
    }
  },
  // ... mais camadas ...
};

#define LAYER_0_BIAS_DIM0 12
#define LAYER_0_BIAS_NDIMS 1

float layer_0_bias[12] = {
  0.12164453,-0.10401021,0.24502701,-0.00012441,-0.21282703,0.14837436,
  // ... mais valores ...
};

#endif // NIR_WEIGHTS_H
```

## Arquivos Gerados no Exemplo

Quando você executa o script completo, os seguintes arquivos são gerados:

- `nir_weights.h`: Pesos principais (formato float, 10 valores por linha)
- `nir_weights_alt.h`: Pesos alternativos (formato weight_t, 8 valores por linha)
- `cpp_output/nir_network.h`: Header com declarações das funções da rede
- `cpp_output/nir_network.cpp`: Implementação das funções da rede
- `cpp_output/main.cpp`: Arquivo main de exemplo

## Compatibilidade

- Compatível com os mesmos formatos de arquivos NIR gerados pelo snnTorch
- Mantém o mesmo padrão de nomenclatura e organização do `export_weights.py`
- Os arquivos `.h` gerados são compatíveis com compiladores C/C++ padrão

## Requisitos

- Python 3.7+
- Bibliotecas: `nir`, `numpy`
- Arquivo `.nir` válido (gerado pelo `export_nir_from_snnTorchNet.py`)

## Exemplo Completo de Uso

```python
#!/usr/bin/env python3
import os
from nir_to_c import export_nir_weights_to_c_header

def main():
    nir_file = "csnn_mnist.nir"
    
    if not os.path.exists(nir_file):
        print(f"Erro: {nir_file} não encontrado!")
        return
    
    # Exportação com configurações personalizadas
    export_nir_weights_to_c_header(
        nir_file=nir_file,
        header_path="network_weights.h",
        ctype="float",
        emit_typedef_if_builtin=False,
        line_wrap=8,
        float_fmt=".6f",
        verbose=True
    )
    
    print("Pesos exportados com sucesso!")

if __name__ == "__main__":
    main()
```