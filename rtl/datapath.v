/* ============================================================
    Datapath da MLP — Registradores, Acumuladores e Argmax
    Projeto: Diagnóstico de Falhas com MLP em Hardware RTL
    CI Digital / CEPEDI
    Alunos: João Vitor e Tatiana Moura
   ============================================================ */
module datapath #(
    parameter DATA_W = 8,     // largura de dados 8 bits
    parameter ACC_W = 24,     // largura do acumulador 24 bits
    parameter N_IN = 4,       // features de entrada
    parameter N_HIDDEN = 8,   // neurônios ocultos
    parameter N_OUT = 5,       // classes de saída
    parameter SCORE_SHIFT = 5
)(
    input wire clk,
    input wire rst_n,

    // Entradas externas de sensor
    input wire signed [DATA_W-1:0] feat_0, // standardized spectral feature / feature selection
    input wire signed [DATA_W-1:0] feat_1, // standardized spectral feature / feature selection
    input wire signed [DATA_W-1:0] feat_2, // standardized spectral feature / feature selection
    input wire signed [DATA_W-1:0] feat_3, // standardized spectral feature / feature selection

    // Controle da FSM
    input wire load_input,              // salvar as 4 features no datapath
    input wire [1:0] input_mux_sel,     // standardized spectral feature / feature selection
    input wire [2:0] hidden_mux_sel,    // seleciona qual dos 8 neurõnios vai para ULA
    input wire store_hidden,            // escrita nos registradores (camada oculta)
    input wire [2:0] store_hidden_idx,  // indice de qual registrador da camada oculta (0..7)
    input wire store_output,            // escrita nos registradores (camada de saída)
    input wire [2:0] store_output_idx,  // indice de qual registrador da camada de saída (0..4)
    input wire do_argmax,               // Quando FSM ordena, datapath busca maoir valor entre as classes

    // Saída para ULA
    output wire signed [DATA_W-1:0] ula_data_in,  //dado de entrada (sensores) do datapath para a ULA

    // Resultado
    input wire signed [DATA_W-1:0] relu_out,       // resultado vindo da ULA para ser guardado no datapath
    input wire signed [ACC_W-1:0] acc_out,         // valor bruto acumulado na 2a camada

    output reg [2:0] class_id,                     // resultado do diagnostico de falahas do chip (0..4)
    output reg signed [DATA_W-1:0] confidence,     // exporta o nível de confiança
    output wire signed [DATA_W-1:0] hidden_to_ula  // dados dos 8 neurônios ocultos para a ULA
);
    // variáveis para indexar laços de repetição
    integer i;

    // Registradores de entrada
    reg signed [DATA_W-1:0] reg_feat [0:N_IN-1]; //array de 4 posições e cada posição com 8 bits

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_feat[0] <= {DATA_W{1'b0}};
            reg_feat[1] <= {DATA_W{1'b0}};
            reg_feat[2] <= {DATA_W{1'b0}};
            reg_feat[3] <= {DATA_W{1'b0}};
        end else if (load_input) begin     //flag para salvar dados
            reg_feat[0] <= feat_0;
            reg_feat[1] <= feat_1;
            reg_feat[2] <= feat_2;
            reg_feat[3] <= feat_3;
        end
    end

    // roteando 1 dado de cada vez para uLA
    assign ula_data_in = reg_feat[input_mux_sel]; // standardized spectral feature / feature selection

    reg signed [DATA_W-1:0] hidden_out [0:N_HIDDEN-1]; //array 8 posições e cada posição 8 bits (memória da camada oculta)

    genvar h;
    generate for (h=0; h<N_HIDDEN; h=h+1) begin: hidden_registers
        always @(posedge clk or negedge rst_n)
            if (!rst_n) hidden_out[h] <= 0;
            else if (store_hidden && store_hidden_idx == h) hidden_out[h] <= relu_out;
    end endgenerate

    // MUX: seleciona saída oculta para ULA (L2)
    wire signed [DATA_W-1:0] hidden_mux_out; //fio que conta de 0..7 (8 neurônios) vindo da FSM
    assign hidden_mux_out = hidden_out[hidden_mux_sel];

    // Acumuladores da camada de saída (L2)
    reg signed [ACC_W-1:0] output_acc [0:N_OUT-1];  //banco de registros para os 5 neurônios finais

    genvar o;
    generate for (o=0; o<N_OUT; o=o+1) begin: output_registers
        always @(posedge clk or negedge rst_n)
            if (!rst_n) output_acc[o] <= 0;
            else if (store_output && store_output_idx == o) output_acc[o] <= acc_out;
    end endgenerate

    // Argmax sobre os 5 acumuladores de saída
    reg signed [ACC_W-1:0] max_val;   //armazena o maior valor encontrado com 24 bits de precisão
    reg [2:0] max_idx;                // armazena o índice (0..4) desse maior valor

    // Variável intermediária para calcular a escala da confiança de forma segura
    reg signed [ACC_W-1:0] conf_calc; //antes de enviar a confiança(8 bits), aplica desloc para restaurar escala (variável auxiliar)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_id <= 3'b000;
            confidence <= {DATA_W{1'b0}};
        end else if (do_argmax) begin
            max_val = output_acc[0]; //primeiramente assume a classe 0 como vencedora
            max_idx = 3'd0;

            // Laço sintetizável para encontrar o maior (argmax)
            for (i = 1; i < N_OUT; i = i + 1) begin
                if ($signed(output_acc[i]) > $signed(max_val)) begin
                    max_val = output_acc[i];
                    max_idx = i[2:0];
                end
            end

            class_id <= max_idx; //atribui o indice final vencedor À porta de saída do chip

            // Ajuste de escala com Saturação
            conf_calc = (SCORE_SHIFT >= 0) ? (max_val >>> SCORE_SHIFT) : (max_val <<< (-SCORE_SHIFT));
            if (conf_calc > 24'sd127) begin
                confidence <= 8'sd127; // Satura no máximo positivo do int8
            end else if (conf_calc < -24'sd128) begin
                confidence <= 8'sh80;
            end else begin
                confidence <= conf_calc[DATA_W-1:0];
            end

        end
    end

    // Porta de saída da camada oculta para top_monitor
    assign hidden_to_ula = hidden_mux_out;

endmodule
