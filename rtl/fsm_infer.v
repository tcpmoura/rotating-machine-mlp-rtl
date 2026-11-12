/* ============================================================
    FSM de Controle da Inferência MLP (Moore)
    Projeto: Diagnóstico de Falhas com MLP em Hardware RTL
    CI Digital / CEPEDI
    Alunos: João Vitor e Tatiana Moura
   ============================================================ */

module fsm_infer #(
    parameter N_IN     = 4, // 4 features dos sensores
    parameter N_HIDDEN = 8, // qtd de neurônios camada oculta
    parameter N_OUT    = 5, // classes finais (saídas)
    parameter ADDR_W   = 7, // largura do endereço da ROM
    parameter DATA_W   = 8  // largura dos dados
)(
    input  wire        clk,
    input  wire        rst_n, // reset ativo em nível baixo
    input  wire        enable, // enable = 1 FSM sai do repouso
    input  wire        tx_done, // sinal da UART avisando transmissão acabou -> IDLE

    // Controles para o datapath
    output reg         load_input, // capturar os dados dos sensores (congela) nos registradores
    output reg  [1:0]  input_mux_sel, // standardized spectral feature / feature selection
    output reg  [2:0]  hidden_mux_sel, // 0 a 8 - rotear os 8 neurônios ocultos
    output reg         store_hidden, // store_hidden = 1 vai habilitar escrita. Manda o datapath salvar resultado da ULA
    output reg  [2:0]  store_hidden_idx, // FSM informa o índice p/ o datapath de qual neurônio da camada oculta (0 a 7)
    output reg         store_output, // habilita a escrita. Manda o datapath salvar a pontuação no registrador da camada de saída (L2)
    output reg  [2:0]  store_output_idx, //informa ao datpath qual indice (0..4) deve ser salvo
    output reg         do_argmax, //manda o datapath executar a varredura e descobrir qual classe teve maior pontuação

    // Controles para a ULA
    output reg  [1:0]  ula_opcode, //informa a ULA qual operação deverá executar
    output reg         layer_sel,  //avisa em qual camada está operando 0-1a camada; 1- 2a cam (top monitor)

    // Controles para a ROM
    output reg         rom_en, //ativa a ROM, diz pra ler o dado no endereço selecionado
    output reg  [ADDR_W-1:0] rom_addr, //FSM informa qual entendeço do peso e bias que ela precisa

    // Controle UART
    output reg         uart_start, //quando =1; ordena UART enviar o diagnóstico

    // Saídas de alerta
    output reg         led_ok, //pino acenderá led verde (condição normal)
    output reg         led_warn, //pino amarelo (falhas de rolamento - não crítico)
    output reg         led_crit, //pino vermelho (desbalanceamento ou desalinhamento - crítico)
    output reg         halt, //corta energia elétrica do motor

    // Estado atual (debug)
    output reg  [2:0]  state_out, //informa em qual estado atual a FSM está (0..6)

    // class_id do datapath
    input  wire [2:0]  class_id //Datapath informa qual a classe venceu, para decidir qual led acender (0..4)
);

    // Estados
    localparam [2:0]
        S_IDLE        = 3'd0, //Repouso
        S_LOAD_INPUT  = 3'd1, //Salvar dados dos sensores e zerar ULA
        S_LAYER1      = 3'd2, //calcular 8 neurônios
        S_LAYER2      = 3'd3, //calcular pontuação das 5 classes finais
        S_CLASSIFY    = 3'd4, //avisa ao datapath para disparar Argmax (Ula acabou cálc)
        S_WAIT_ARGMAX = 3'd5, //espera 1 ciclo para datapath para eleger vencedor
        S_TRANSMIT    = 3'd6; //acende LED e dispara UART

    // Endereços Base da ROM
    localparam [ADDR_W-1:0]
        ADDR_W1_BASE = 7'h00, //pesos da camada oculta (4x8)
        ADDR_B1_BASE = 7'h20, //biases camada oculta (8)
        ADDR_W2_BASE = 7'h28, //pesos da camada de saída (8x5)
        ADDR_B2_BASE = 7'h50; //biases da camada de saída (5)

    // Opcodes da ULA
    localparam [1:0]
        OP_NOP   = 2'b00, //mantém valor atual
        OP_MAC   = 2'b01, //multiplicação
        OP_ADD   = 2'b10, //soma
        OP_CLEAR = 2'b11; //esvaziar a ula

    reg [2:0] state, next_state; //state - onde a fsm está agora; next state..
    reg uart_sent; // One start pulse per inference, held until returning to IDLE.

    // Contadores de iteração
    reg [1:0] input_cnt;    // 0..3 varrerá as 4 features
    reg [2:0] neuron_cnt;   // 0..7 varrerá os 8 neurônios L1
    reg [2:0] hidden_cnt;   // 0..7 varrerá os 8 neurônios calculados anteriormente(entradas da L2)
    reg [2:0] output_cnt;   // 0..4  qual das 5 classes estamos calculando

    // Fases do Neurônio
    reg       bias_phase;   // 1 = terminaram as multiplicações de um neurônio, manda ler o bias da rom
    reg       wait_phase;   // 1 = Aguarda 1 clock, para estabilizar a saída da ULA (terminar de somar bias)
    reg       store_phase;  // 1 = Salva resultado do neurônio no datapath e limpa ULA

    // Registro de estado
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // Lógica de transição dos estados - combinacional
    always @(*) begin
        next_state = state; //se não houver mudança, próx estado é o atual (latch)
        case (state)
            S_IDLE:
                if (enable) next_state = S_LOAD_INPUT;
            S_LOAD_INPUT:
                next_state = S_LAYER1;
            S_LAYER1:
                if (neuron_cnt == N_HIDDEN-1 && store_phase) //varreu os 8 neuronios e etapa de armazenamento está ativa
                    next_state = S_LAYER2;
            S_LAYER2:
                if (output_cnt == N_OUT-1 && store_phase)
                    next_state = S_CLASSIFY;
            S_CLASSIFY:
                next_state = S_WAIT_ARGMAX; //1 ciclo para comparadores no datapath
            S_WAIT_ARGMAX:
                next_state = S_TRANSMIT; //1 ciclo para dar tempo do datapath conseguir encontrar resultado
            S_TRANSMIT:
                if (tx_done) next_state = S_IDLE; //quando UART envia tx_done=1
            default:
                next_state = S_IDLE;
        endcase
    end

    // Lógica de saída + contadores
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_cnt        <= 2'd0; //zera sinais de controle
            neuron_cnt       <= 3'd0;
            hidden_cnt       <= 3'd0;
            output_cnt       <= 3'd0;
            bias_phase       <= 1'b0;
            wait_phase       <= 1'b0;
            store_phase      <= 1'b0;

            load_input       <= 1'b0;
            input_mux_sel    <= 2'd0;
            hidden_mux_sel   <= 3'd0;
            store_hidden     <= 1'b0;
            store_hidden_idx <= 3'd0;
            store_output     <= 1'b0;
            store_output_idx <= 3'd0;
            do_argmax        <= 1'b0;
            ula_opcode       <= OP_NOP;
            layer_sel        <= 1'b0;
            rom_en           <= 1'b0;
            rom_addr         <= {ADDR_W{1'b0}};
            uart_start       <= 1'b0;
            uart_sent        <= 1'b0;
            led_ok           <= 1'b0;
            led_warn         <= 1'b0;
            led_crit         <= 1'b0;
            halt             <= 1'b0;
            state_out        <= S_IDLE;
        end else begin
            // Default: zerar registradores para não guardarem lixo de ciclos anteriores
            load_input       <= 1'b0;
            store_hidden     <= 1'b0;
            store_output     <= 1'b0;
            do_argmax        <= 1'b0;
            ula_opcode       <= OP_NOP;
            uart_start       <= 1'b0;
            rom_en           <= 1'b0;
            state_out        <= state;

            case (state)
                S_IDLE: begin
                    uart_sent <= 1'b0;
                    input_cnt   <= 2'd0; //todos os indices de contadores zerados -> lixo de memória
                    neuron_cnt  <= 3'd0;
                    hidden_cnt  <= 3'd0;
                    output_cnt  <= 3'd0;
                    bias_phase  <= 1'b0; //registradoresw de fase zerados
                    wait_phase  <= 1'b0;
                    store_phase <= 1'b0;
                end

                S_LOAD_INPUT: begin           //salvar dados nos sensores e zerar a ula
                    load_input  <= 1'b1;      //ativa captura dos sensores
                    layer_sel   <= 1'b0;
                    ula_opcode  <= OP_CLEAR; // Zera a ULA para iniciar a L1
                    neuron_cnt  <= 3'd0;     // varrer 8 neuronio
                    input_cnt   <= 2'd0;     // varrer 4 features
                    bias_phase  <= 1'b0;
                    wait_phase  <= 1'b0;
                    store_phase <= 1'b0;
                end

                S_LAYER1: begin
                    layer_sel <= 1'b0;
                    if (!bias_phase && !wait_phase && !store_phase) begin
                        // Ciclo de pesos (MAC)
                        rom_en        <= 1'b1;
                        rom_addr      <= ADDR_W1_BASE + {1'b0, neuron_cnt, input_cnt};
                        input_mux_sel <= input_cnt; // standardized spectral feature / feature selection
                        ula_opcode    <= OP_MAC; //multiplicação

                        if (input_cnt == N_IN-1) begin //chegou ao ultimo sensor de entrada
                            input_cnt  <= 2'd0;
                            bias_phase <= 1'b1; // Inicia fase do bias
                        end else begin
                            input_cnt <= input_cnt + 1'b1;
                        end
                    end else if (bias_phase) begin
                        // Manda endereço do Bias e habilita ULA
                        rom_en        <= 1'b1;
                        rom_addr      <= ADDR_B1_BASE + {4'b0, neuron_cnt};
                        ula_opcode    <= OP_MAC;

                        bias_phase    <= 1'b0;
                        wait_phase    <= 1'b1; // Aguarda ULA somar
                    end else if (wait_phase) begin
                        // Aguarda 1 ciclo para soma do Bias ser registrada
                        wait_phase    <= 1'b0;
                        store_phase   <= 1'b1; // Prepara para salvar
                    end else if (store_phase) begin
                        // Salva o resultado final no Datapath e limpa a ULA
                        store_hidden     <= 1'b1;
                        store_hidden_idx <= neuron_cnt;
                        ula_opcode       <= OP_CLEAR;
                        store_phase      <= 1'b0;

                        if (neuron_cnt == N_HIDDEN-1) begin //os 8 neurônios foram processados?
                            neuron_cnt <= 3'd0;
                            hidden_cnt <= 3'd0;
                            output_cnt <= 3'd0;
                        end else begin
                            neuron_cnt <= neuron_cnt + 1'b1;
                        end
                    end
                end

                S_LAYER2: begin
                    layer_sel <= 1'b1;
                    if (!bias_phase && !wait_phase && !store_phase) begin
                        rom_en         <= 1'b1;
                        rom_addr       <= ADDR_W2_BASE
                                        + {1'b0, output_cnt, 3'b000}
                                        + {4'b0, hidden_cnt};
                        hidden_mux_sel <= hidden_cnt; //valor do contador da fsm é aplicado ao mux do datapath
                        ula_opcode     <= OP_MAC;

                        if (hidden_cnt == N_HIDDEN-1) begin
                            hidden_cnt <= 3'd0;
                            bias_phase <= 1'b1;
                        end else begin
                            hidden_cnt <= hidden_cnt + 1'b1;
                        end
                    end else if (bias_phase) begin
                        rom_en   <= 1'b1;
                        rom_addr <= ADDR_B2_BASE + {4'b0, output_cnt};
                        ula_opcode <= OP_MAC; //OBS

                        bias_phase  <= 1'b0;
                        wait_phase  <= 1'b1;
                    end else if (wait_phase) begin
                        wait_phase  <= 1'b0;
                        store_phase <= 1'b1;
                    end else if (store_phase) begin
                        store_output     <= 1'b1;
                        store_output_idx <= output_cnt;
                        ula_opcode       <= OP_CLEAR;
                        store_phase      <= 1'b0;

                        if (output_cnt == N_OUT-1) begin
                            output_cnt <= 3'd0;
                        end else begin
                            output_cnt <= output_cnt + 1'b1;
                        end
                    end
                end

                S_CLASSIFY: begin
                    do_argmax <= 1'b1; // Dispara o argmax para que o datapath possa comparar as 5 classes
                end

                S_WAIT_ARGMAX: begin
                    // Apenas aguarda 1 ciclo de clock
                end

                S_TRANSMIT: begin
                    if (!uart_sent) begin
                        uart_sent <= 1'b1;
                        uart_start <= 1'b1; //aciona UART
                        case (class_id)
                            3'd0: begin
                                led_ok   <= 1'b1; led_warn <= 1'b0; led_crit <= 1'b0; halt <= 1'b0;
                            end
                            3'd1, 3'd2: begin
                                led_ok   <= 1'b0; led_warn <= 1'b1; led_crit <= 1'b0; halt <= 1'b0;
                            end
                            3'd3, 3'd4: begin
                                led_ok   <= 1'b0; led_warn <= 1'b0; led_crit <= 1'b1; halt <= 1'b1;
                            end
                            default: begin
                                led_ok   <= 1'b1; led_warn <= 1'b0; led_crit <= 1'b0; halt <= 1'b0;
                            end
                        endcase
                    end
                end
            endcase
        end
    end

endmodule
