/* ============================================================
    Unidade Lógica e Aritmética - ULA
    Projeto: Diagnóstico de Falhas com MLP em Hardware RTL
    CI Digital / CEPEDI
    Alunos: João Vitor e Tatiana Moura

    Operações (opcode):
    00: NOP   - Mantém o valor atual do acumulador
    01: MAC   - Multiplica (weight * data_in) e acumula
    10: ADD   - Apenas soma a entrada (weight) ao acumulador
    11: CLEAR - Zera o acumulador
 =============================================================== */

module ula_neural #(
    parameter DATA_W = 8, // largura de dados 8 bits
    parameter ACC_W  = 24, // largura do acumulador 24 bits
    parameter HIDDEN_SHIFT = 5
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Barramento de controle da ULA
    input  wire [1:0]            opcode,     //Operação: 00 NOP, 01 MAC, 10 ADD e 11 CLEAR

    input  wire signed [DATA_W-1:0] weight,  //recebe pesos e biases
    input  wire signed [DATA_W-1:0] data_in, //recebe features sensores (L1); recebe os valores dos neurônios intermediários (L2)

    output reg  signed [ACC_W-1:0]  acc_out, //exporte valor bruto do acumulador -> datapath
    output wire signed [DATA_W-1:0] relu_out //resultado final do neurônio
);

    // Produto com Extensão de Sinal
    wire signed [ACC_W-1:0] product = $signed(weight) * $signed(data_in);

    // Acumulador Síncrono controlado pelo decodificador de instruções
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= $signed({ACC_W{1'b0}});
        end else begin
            case (opcode)
                2'b00: acc_out <= acc_out;                                                         // NOP
                2'b01: acc_out <= acc_out + product;                                               // MAC
                2'b10: acc_out <= acc_out + $signed({ {ACC_W-DATA_W{weight[DATA_W-1]}}, weight }); // ADD (Soma direta com extensão explícita): weight[DATA_W-1] é o MSB do peso (bit de sinal)...ACC_W-DATA_W = 24-8 (quantos bits faltam para 24) = 16  ... {16{weight[7]}}, ou seja, pega o bit de sinal e replica 16 vezes e junta com os 8 bits originais do peso
                2'b11: acc_out <= $signed({ACC_W{1'b0}});                                          // CLEAR
                default: acc_out <= acc_out;
            endcase
        end
    end

    // ReLU
    wire signed [ACC_W-1:0] relu_full = (acc_out[ACC_W-1]) ? $signed({ACC_W{1'b0}}) : acc_out;

    // Adiciona 16 para emular o round() do Python e desloca 5 bits p/ a direita (divisão por 32)
    localparam signed [ACC_W-1:0] ROUND_HALF = 1 << (HIDDEN_SHIFT-1);
    wire signed [ACC_W-1:0] relu_shifted = (relu_full + ROUND_HALF) >>> HIDDEN_SHIFT;

    // Saturação em +127
    assign relu_out = (relu_shifted > 24'sd127) ? 8'sd127 : $signed(relu_shifted[DATA_W-1:0]);

endmodule
