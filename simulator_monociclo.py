
import sys

# --- Configurações Globais ---
DATA_WIDTH = 32
REGISTER_COUNT = 32
MEMORY_SIZE_WORDS = 2**16 # 65536 palavras (2^16 endereços)
MASK = 0xFFFFFFFF

# --- Estado do Processador ---
PC = 0
REGISTERS = [0] * REGISTER_COUNT
MEMORY = [0] * MEMORY_SIZE_WORDS
FLAGS = {"N": 0, "Z": 0, "C": 0, "V": 0}

# --- Estado da Instrução Atual (Mantido entre ciclos) ---
CURRENT_INSTR = {
    "PC_START": 0,
    "INSTR_WORD": 0,
    "PC_NEXT_DEFAULT": 0,
    "HALT": False,
    "STAGE": 0, # 0: IF, 1: ID, 2: EX/MEM, 3: WB
    # ID Stage Outputs
    "RA_IDX": 0, "RB_IDX": 0, "RC_IDX": 0,
    "READ_DATA_1": 0, "READ_DATA_2": 0,
    "IMM": 0, "IMM_BRANCH": 0, "JUMP_TARGET_ADDR": 0,
    "REG_WRITE": 0, "MEM_READ": 0, "MEM_WRITE": 0, "MEM_TO_REG": 0,
    "ALU_SRC": 0, "BRANCH": 0, "JUMP": 0,
    "WRITE_REG_ADDR": 0, "ALU_OP": "NOP", "INSTR_NAME": "NOP",
    # EX/MEM Stage Outputs
    "ALU_RESULT": 0,
    "BRANCH_TAKEN": False,
    # MEM Stage Outputs
    "MEM_READ_DATA": 0,
    "DATA_TO_WRITE_MEM": 0,
}

# --- Dicionário de Instruções ---
INSTRUCTIONS = {
    # NOP
    "00000000": {"name": "NOP", "format": "R", "ALUOp": "NOP", "updates_flags": False},
    # Halt
    "11111111": {"name": "HALT", "format": "R", "ALUOp": "HALT", "updates_flags": False},

    # Aritméticas (R-Type)
    "00000001": {"name": "ADD", "format": "R", "ALUOp": "ADD", "updates_flags": True},
    "00000010": {"name": "SUB", "format": "R", "ALUOp": "SUB", "updates_flags": True},
    "00011000": {"name": "MULT", "format": "R", "ALUOp": "MULT", "updates_flags": True}, # NOVO OPCODE
    "00011001": {"name": "DIV", "format": "R", "ALUOp": "DIV", "updates_flags": True}, # NOVO OPCODE
    "00011010": {"name": "MOD", "format": "R", "ALUOp": "MOD", "updates_flags": True}, # NOVO OPCODE

    # Lógicas (R-Type)
    "00000011": {"name": "ZER", "format": "R_RC", "ALUOp": "ZER", "updates_flags": True}, # Corrigido: ZER (flags=True)
    "00000100": {"name": "XOR", "format": "R", "ALUOp": "XOR", "updates_flags": True},
    "00000101": {"name": "OR", "format": "R", "ALUOp": "OR", "updates_flags": True},
    "00000110": {"name": "NOT", "format": "R_RA_RC", "ALUOp": "NOT", "updates_flags": True},
    "00000111": {"name": "AND", "format": "R", "ALUOp": "AND", "updates_flags": True},

    # Shifts (R-Type)
    "00001000": {"name": "ASL", "format": "R", "ALUOp": "ASL", "updates_flags": True},
    "00001001": {"name": "ASR", "format": "R", "ALUOp": "ASR", "updates_flags": True},
    "00001010": {"name": "LSL", "format": "R", "ALUOp": "LSL", "updates_flags": True},
    "00001011": {"name": "LSR", "format": "R", "ALUOp": "LSR", "updates_flags": True},

    # Movimentação de Dados (R-Type)
    "00001100": {"name": "COPY", "format": "R_RA_RC", "ALUOp": "COPY", "updates_flags": False},

    # Comparação (R-Type)
    "00010111": {"name": "SLT", "format": "R", "ALUOp": "SLT", "updates_flags": False}, # NOVO OPCODE

    # Carga de Constantes (I-Type)
    "00001110": {"name": "LUI", "format": "I_CONST16", "ALUOp": "LUI", "updates_flags": False},
    "00001111": {"name": "LLI", "format": "I_CONST16", "ALUOp": "LLI", "updates_flags": False},

    # Acesso à Memória (I-Type)
    "00010000": {"name": "LW", "format": "I_MEM", "ALUOp": "ADD", "updates_flags": False},
    "00010001": {"name": "SW", "format": "I_MEM", "ALUOp": "ADD", "updates_flags": False},

    # Desvios Incondicionais
    "00010010": {"name": "JAL", "format": "J", "ALUOp": "JAL", "updates_flags": False},
    "00010011": {"name": "JR", "format": "R_RC", "ALUOp": "NOP", "updates_flags": False},

    # Desvios Condicionais (R-Type com Endereçamento)
    "00010100": {"name": "BEQ", "format": "R_END", "ALUOp": "SUB", "updates_flags": True},
    "00010101": {"name": "BNE", "format": "R_END", "ALUOp": "SUB", "updates_flags": True},

    # Desvios Incondicionais
    "00010110": {"name": "JMP", "format": "J", "ALUOp": "NOP", "updates_flags": False},

    # I-Type Adicionais
    "10000001": {"name": "ADDI", "format": "I_CONST16", "ALUOp": "ADD", "updates_flags": True},
    "10000010": {"name": "SUBI", "format": "I_CONST16", "ALUOp": "SUB", "updates_flags": True},
    "10001100": {"name": "ANDI", "format": "I_CONST16", "ALUOp": "AND", "updates_flags": True},
    "10000100": {"name": "ORI", "format": "I_CONST16", "ALUOp": "OR", "updates_flags": True},
}

# --- Variáveis de Simulação ---
cycle = 0
REG_CHANGES = []
MEM_CHANGES = []

# --- Funções Auxiliares ---
def to_signed(value, bits):
    """Converte um valor para inteiro com sinal."""
    sign_mask = 1 << (bits - 1)
    return (value & (sign_mask - 1)) - (value & sign_mask)

def get_reg_index(reg_field):
    """Retorna o índice do registrador a partir do campo da instrução."""
    return reg_field

def read_memory(address):
    """Lê uma palavra da memória."""
    if 0 <= address < MEMORY_SIZE_WORDS:
        return MEMORY[address]
    else:
        raise IndexError(f"Endereço de memória inválido: {address}")

def write_memory(address, data):
    """Escreve uma palavra na memória."""
    global MEM_CHANGES
    if 0 <= address < MEMORY_SIZE_WORDS:
        old_value = MEMORY[address]
        MEMORY[address] = data & MASK
        if old_value != MEMORY[address]:
            MEM_CHANGES.append((address, old_value, MEMORY[address]))
    else:
        raise IndexError(f"Endereço de memória inválido: {address}")

def update_flags(result, op1, op2, alu_op, opcode):
    """Atualiza as flags N, Z, C, V com base no resultado da ALU."""
    result_signed = to_signed(result, DATA_WIDTH)
    op1_signed = to_signed(op1, DATA_WIDTH)
    op2_signed = to_signed(op2, DATA_WIDTH)

    # Flag N (Negative)
    FLAGS["N"] = 1 if result_signed < 0 else 0

    # Flag Z (Zero)
    FLAGS["Z"] = 1 if (result & MASK) == 0 else 0

    # Flags C (Carry) e V (Overflow)
    if alu_op in ["ADD", "ADDI"]:
        # Carry (C): C=1 se houver carry-out do MSB
        FLAGS["C"] = 1 if (op1 + op2) > MASK else 0
        # Overflow (V): Sinais iguais e resultado com sinal diferente
        FLAGS["V"] = 1 if (op1_signed >= 0 and op2_signed >= 0 and result_signed < 0) or \
                          (op1_signed < 0 and op2_signed < 0 and result_signed >= 0) else 0
    elif alu_op in ["SUB", "SUBI"] or opcode in ["00010100", "00010101"]: # SUB, SUBI, BEQ, BNE
        # Carry (C): C=1 se não houver borrow (op1 >= op2 unsigned)
        FLAGS["C"] = 1 if (op1 & MASK) >= (op2 & MASK) else 0
        # Overflow (V): Sinais diferentes e resultado com sinal diferente
        FLAGS["V"] = 1 if (op1_signed >= 0 and op2_signed < 0 and result_signed < 0) or \
                          (op1_signed < 0 and op2_signed >= 0 and result_signed >= 0) else 0
    elif alu_op in ["ASL", "LSL"]:
        # Carry (C): C é o último bit que saiu pela esquerda
        FLAGS["C"] = (op1 >> (DATA_WIDTH - op2)) & 1 if op2 > 0 else 0
    elif alu_op in ["ASR", "LSR"]:
        # Carry (C): C é o último bit que saiu pela direita
        FLAGS["C"] = (op1 >> (op2 - 1)) & 1 if op2 > 0 else 0

    return result

# --- Funções de Estágios (Monociclo de 4 Ciclos) ---

def IF_stage():
    """Instruction Fetch: Busca a instrução e calcula o PC da próxima instrução."""
    global PC, CURRENT_INSTR, cycle

    CURRENT_INSTR["PC_START"] = PC
    CURRENT_INSTR["PC_NEXT_DEFAULT"] = PC + 1
    CURRENT_INSTR["STAGE"] = 0

    try:
        instruction = read_memory(PC)
    except IndexError:
        print(f"ERRO: Tentativa de buscar instrução em endereço inválido: {PC}")
        instruction = 0xFFFFFFFF # HALT

    CURRENT_INSTR["INSTR_WORD"] = instruction

    # O PC é atualizado apenas no WB_stage. O valor de PC_NEXT_DEFAULT é apenas
    # o endereço da próxima instrução sequencial. O PC atual (PC_START) é mantido
    # para ser usado no cálculo de desvio no WB_stage.
    # O PC global NÃO é atualizado aqui.

def ID_stage():
    """Instruction Decode: Decodifica a instrução e lê registradores."""
    global REGISTERS, CURRENT_INSTR

    CURRENT_INSTR["STAGE"] = 1
    instruction_word = CURRENT_INSTR["INSTR_WORD"]

    if instruction_word == 0 or CURRENT_INSTR["INSTR_WORD"] == 0xFFFFFFFF:
        return # NOP ou HALT

    instruction_bin = format(instruction_word, f'0{DATA_WIDTH}b')
    opcode = instruction_bin[:8]

    instr_info = INSTRUCTIONS.get(opcode, {"name": "UNKNOWN", "format": "R", "ALUOp": "NOP"})
    instr_name = instr_info["name"]
    instr_format = instr_info["format"]
    alu_op_initial = instr_info["ALUOp"]

    # Campos de Registradores
    ra_field = int(instruction_bin[8:16], 2)
    rb_field = int(instruction_bin[16:24], 2)
    rc_field = int(instruction_bin[24:32], 2)

    ra_idx = get_reg_index(ra_field)
    rb_idx = get_reg_index(rb_field)
    rc_idx = get_reg_index(rc_field)

    # Leitura de Registradores
    read_data_1 = REGISTERS[ra_idx]
    read_data_2 = REGISTERS[rb_idx]

    # Campos Imediatos/Endereços
    imm_const16 = int(instruction_bin[16:32], 2)
    imm_branch = to_signed(int(instruction_bin[8:32], 2), 24)
    jump_target_addr = int(instruction_bin[8:32], 2)

    # Sinais de Controle (Inicialização)
    reg_write = 0
    mem_read = 0
    mem_write = 0
    mem_to_reg = 0
    alu_src = 0  # 0: Read_Data_2, 1: Imm
    branch = 0
    jump = 0
    write_reg_addr = rc_idx
    alu_op = alu_op_initial
    imm = 0
    data_to_write_mem = 0

    # Decodificação específica
    if instr_format == "R":
        reg_write = 1
    elif instr_format == "R_RA_RC":
        reg_write = 1
    elif instr_format == "R_RC":
        if instr_name == "ZER":
            reg_write = 1
        elif instr_name == "JR":
            jump = 1
            jump_target_addr = read_data_2 # Endereço de pulo é R[rb]
            write_reg_addr = 0
    elif instr_format == "I_CONST16":
        reg_write = 1
        alu_src = 1
        imm = imm_const16
    elif instr_format == "I_MEM":
        # Endereço é o conteúdo de R[ra]
        if instr_name == "LW":
            mem_read = 1
            reg_write = 1
            mem_to_reg = 1
        elif instr_name == "SW":
            mem_write = 1
            data_to_write_mem = REGISTERS[rc_idx] # Dado a ser escrito é R[rc]
            write_reg_addr = 0
    elif instr_format == "R_END":
        branch = 1
        write_reg_addr = 0
    elif instr_format == "J":
        jump = 1
        if instr_name == "JAL":
            reg_write = 1
            write_reg_addr = 31 # R31
        else: # JMP
            write_reg_addr = 0

    CURRENT_INSTR.update({
        "RA_IDX": ra_idx, "RB_IDX": rb_idx, "RC_IDX": rc_idx,
        "READ_DATA_1": read_data_1, "READ_DATA_2": read_data_2,
        "IMM": imm, "IMM_BRANCH": imm_branch, "JUMP_TARGET_ADDR": jump_target_addr,
        "REG_WRITE": reg_write, "MEM_READ": mem_read, "MEM_WRITE": mem_write, "MEM_TO_REG": mem_to_reg,
        "ALU_SRC": alu_src, "BRANCH": branch, "JUMP": jump,
        "WRITE_REG_ADDR": write_reg_addr, "ALU_OP": alu_op, "INSTR_NAME": instr_name,
        "DATA_TO_WRITE_MEM": data_to_write_mem,
    })

def EX_MEM_stage():
    """Execute and Memory: Executa a ALU e acessa a memória (se necessário)."""
    global FLAGS, CURRENT_INSTR

    CURRENT_INSTR["STAGE"] = 2

    if CURRENT_INSTR["INSTR_WORD"] == 0 or CURRENT_INSTR["INSTR_WORD"] == 0xFFFFFFFF:
        return # NOP ou HALT

    # --- EX Stage (Execução) ---
    op1 = CURRENT_INSTR["READ_DATA_1"]
    op2 = CURRENT_INSTR["READ_DATA_2"] if CURRENT_INSTR["ALU_SRC"] == 0 else CURRENT_INSTR["IMM"]
    alu_op = CURRENT_INSTR["ALU_OP"]
    instr_name = CURRENT_INSTR["INSTR_NAME"]
    rc_idx = CURRENT_INSTR["RC_IDX"]
    opcode = format(CURRENT_INSTR["INSTR_WORD"], f'0{DATA_WIDTH}b')[:8]

    alu_result = 0
    if alu_op == "ADD":
        alu_result = op1 + op2
    elif alu_op == "SUB":
        alu_result = op1 - op2
    elif alu_op == "ZER":
        alu_result = 0
    elif alu_op == "XOR":
        alu_result = op1 ^ op2
    elif alu_op == "OR":
        alu_result = op1 | op2
    elif alu_op == "NOT":
        alu_result = ~op1
    elif alu_op == "AND":
        alu_result = op1 & op2
    elif alu_op == "ASL":
        alu_result = op1 << op2
    elif alu_op == "ASR":
        alu_result = to_signed(op1, DATA_WIDTH) >> op2
    elif alu_op == "LSL":
        alu_result = op1 << op2
    elif alu_op == "LSR":
        alu_result = op1 >> op2
    elif alu_op == "COPY":
        alu_result = op1
    elif alu_op == "LUI":
        alu_result = (CURRENT_INSTR["IMM"] << 16) | (REGISTERS[rc_idx] & 0x0000FFFF)
    elif alu_op == "LLI":
        alu_result = CURRENT_INSTR["IMM"] | (REGISTERS[rc_idx] & 0xFFFF0000)
    elif alu_op == "MULT":
        alu_result = op1 * op2
    elif alu_op == "DIV":
        alu_result = op1 // op2 if op2 != 0 else 0
    elif alu_op == "MOD":
        alu_result = op1 % op2 if op2 != 0 else 0
    elif alu_op == "SLT":
        alu_result = 1 if to_signed(op1, DATA_WIDTH) < to_signed(op2, DATA_WIDTH) else 0
    elif alu_op == "JAL":
        alu_result = CURRENT_INSTR["PC_NEXT_DEFAULT"] # R31 = PC + 1

    if INSTRUCTIONS[opcode].get("updates_flags", False):
        # Para ZER, as flags são fixas: neg=0, zero=1, carry=0, overflow=0
        if alu_op == "ZER":
            FLAGS["N"] = 0
            FLAGS["Z"] = 1
            FLAGS["C"] = 0
            FLAGS["V"] = 0
        else:
            update_flags(alu_result, op1, op2, alu_op, opcode)

    # Lógica de Branch
    branch_taken = False
    if CURRENT_INSTR["BRANCH"]:
        if instr_name == "BEQ" and FLAGS["Z"] == 1:
            branch_taken = True
        elif instr_name == "BNE" and FLAGS["Z"] == 0:
            branch_taken = True

    CURRENT_INSTR["ALU_RESULT"] = alu_result
    CURRENT_INSTR["BRANCH_TAKEN"] = branch_taken

    # --- MEM Stage (Memória) ---
    mem_read_data = 0
    if CURRENT_INSTR["MEM_READ"]: # LW
        mem_addr = CURRENT_INSTR["READ_DATA_1"] # Endereço é R[ra]
        try:
            mem_read_data = read_memory(mem_addr)
        except IndexError as e:
            print(f"ERRO: {e}")
            mem_read_data = 0
    elif CURRENT_INSTR["MEM_WRITE"]: # SW
        mem_addr = CURRENT_INSTR["READ_DATA_1"] # Endereço é R[ra]
        data_to_write = CURRENT_INSTR["DATA_TO_WRITE_MEM"] # Dado a ser escrito é R[rc]
        try:
            write_memory(mem_addr, data_to_write)
        except IndexError as e:
            print(f"ERRO: {e}")

    CURRENT_INSTR["MEM_READ_DATA"] = mem_read_data

def WB_stage():
    """Write Back: Escreve o resultado no registrador e atualiza o PC."""
    global PC, REGISTERS, CURRENT_INSTR, REG_CHANGES

    CURRENT_INSTR["STAGE"] = 3

    # Se for HALT, a instrução não faz nada, mas o ciclo deve ser completado.
    if CURRENT_INSTR["INSTR_WORD"] == 0:
        # NOP não faz nada
        PC = CURRENT_INSTR["PC_NEXT_DEFAULT"]
    elif CURRENT_INSTR["INSTR_WORD"] == 0xFFFFFFFF:
        # HALT não faz nada, mas o PC deve ser atualizado para o endereço da instrução HALT + 1
        # para que o loop principal possa parar no próximo ciclo de IF.
        PC = CURRENT_INSTR["PC_NEXT_DEFAULT"]
    else:
        # Escrita em Registrador
        if CURRENT_INSTR["REG_WRITE"] and CURRENT_INSTR["WRITE_REG_ADDR"] != 0:
            write_reg_addr = CURRENT_INSTR["WRITE_REG_ADDR"]
            write_data = CURRENT_INSTR["MEM_READ_DATA"] if CURRENT_INSTR["MEM_TO_REG"] else CURRENT_INSTR["ALU_RESULT"]

            old_value = REGISTERS[write_reg_addr]
            REGISTERS[write_reg_addr] = write_data & MASK
            if old_value != REGISTERS[write_reg_addr]:
                REG_CHANGES.append((write_reg_addr, old_value, REGISTERS[write_reg_addr]))

        # Atualização do PC (para instruções que não são HALT)
        if CURRENT_INSTR["JUMP"]:
            PC = CURRENT_INSTR["JUMP_TARGET_ADDR"]
        elif CURRENT_INSTR["BRANCH"] and CURRENT_INSTR["BRANCH_TAKEN"]:
            PC = CURRENT_INSTR["PC_START"] + CURRENT_INSTR["IMM_BRANCH"]
        else:
            PC = CURRENT_INSTR["PC_NEXT_DEFAULT"] # PC = PC + 1 (sequencial)
    
    # R0 é sempre 0
    REGISTERS[0] = 0

# --- Loop de Simulação ---
def run_simulation(max_cycles=1000):
    """Roda a simulação por um número máximo de ciclos."""
    global cycle, PC, CURRENT_INSTR

    print("--- Início da Simulação Monociclo de 4 Ciclos ---")

    while cycle < max_cycles:
        REG_CHANGES.clear()
        MEM_CHANGES.clear()

        # 1. IF Stage
        IF_stage()
        cycle += 1
        print(f"\n--- Ciclo {cycle} (PC={CURRENT_INSTR['PC_START']}) ---")
        print_stage_info("IF", CURRENT_INSTR["INSTR_WORD"], CURRENT_INSTR["PC_START"])

        # Verifica HALT APÓS o IF, mas continua os ciclos
        is_halt = (CURRENT_INSTR["INSTR_WORD"] == 0xFFFFFFFF)

        # 2. ID Stage
        ID_stage()
        cycle += 1
        print(f"\n--- Ciclo {cycle} (PC={CURRENT_INSTR['PC_START']}) ---")
        print_stage_info("ID", CURRENT_INSTR["INSTR_WORD"], CURRENT_INSTR["PC_START"])

        # 3. EX/MEM Stage
        EX_MEM_stage()
        cycle += 1
        print(f"\n--- Ciclo {cycle} (PC={CURRENT_INSTR['PC_START']}) ---")
        print_stage_info("EX/MEM", CURRENT_INSTR["INSTR_WORD"], CURRENT_INSTR["PC_START"])
        
        # Problema 1.2: Mudanças de Memória (SW) devem aparecer aqui
        print_mem_changes()

        # 4. WB Stage
        WB_stage()
        cycle += 1
        print(f"\n--- Ciclo {cycle} (PC={CURRENT_INSTR['PC_START']}) ---")
        print_stage_info("WB", CURRENT_INSTR["INSTR_WORD"], CURRENT_INSTR["PC_START"])
        
        # Problema 1.2: Mudanças de Registradores (R-Type, LW, JAL) devem aparecer aqui
        print_reg_changes()

        if is_halt:
            # A instrução HALT foi executada e o PC foi atualizado no WB_stage.
            # O loop deve parar após o WB_stage.
            print("\n--- HALT detectado. Fim da Simulação ---")
            break

        if cycle >= max_cycles:
            print("\n--- Limite de ciclos atingido. Fim da Simulação ---")
            break

    print_final_state()

def print_stage_info(stage, instruction_word, pc_start):
    """Imprime as informações do estágio, incluindo PC e IR (Problema 1.1)."""
    
    instr_hex = f"0x{instruction_word:08X}"
    
    if instruction_word == 0:
        instr_name = "NOP"
    elif instruction_word == 0xFFFFFFFF:
        instr_name = "HALT"
    else:
        instruction_bin = format(instruction_word, f'0{DATA_WIDTH}b')
        opcode = instruction_bin[:8]
        instr_info = INSTRUCTIONS.get(opcode, {"name": "UNKNOWN"})
        instr_name = instr_info['name']

    # Formato de saída exigido: PC=0x00000000, IR=0x00000000, Estágio: NOME_INSTRUCAO
    output = f"PC=0x{pc_start:08X}, IR={instr_hex}, Estágio {stage}: {instr_name}"
    
    # Adiciona detalhes específicos para o estágio IF
    if stage == "IF":
        output += f" (PC_NEXT=0x{pc_start + 1:08X})"
        
    print(output)

# --- Funções de Impressão ---
def print_reg_changes():
    """Imprime as mudanças nos registradores."""
    if REG_CHANGES:
        print("Alterações de Registradores:")
        for reg_idx, old_val, new_val in REG_CHANGES:
            print(f"  R{reg_idx}: {old_val} -> {new_val}")
    else:
        print("Sem alterações de registradores.")

def print_mem_changes():
    """Imprime as mudanças na memória."""
    if MEM_CHANGES:
        print("Alterações de Memória:")
        for addr, old_val, new_val in MEM_CHANGES:
            print(f"  MEM[{addr}]: {old_val} -> {new_val}")
    else:
        print("Sem alterações de memória.")

def print_final_state():
    """Imprime o estado final dos registradores e da memória."""
    print("\n--- Estado Final ---")
    print("Registradores:")
    for i in range(REGISTER_COUNT):
        if REGISTERS[i] != 0:
            print(f"  R{i}: {REGISTERS[i]} (0x{REGISTERS[i]:08X})")
    print("\nMemória (posições não nulas):")
    for i in range(MEMORY_SIZE_WORDS):
        if MEMORY[i] != 0:
            print(f"  MEM[{i}]: {MEMORY[i]} (0x{MEMORY[i]:08X})")

# --- Carregamento do Programa ---
def load_program_from_file(filepath):
    """Carrega um programa em binário para a memória, suportando a diretiva 'address'."""
    try:
        with open(filepath, 'r') as f:
            address = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                
                if parts[0].lower() == "address":
                    if len(parts) == 2:
                        try:
                            # O endereço é dado em decimal ou hexadecimal (0x...)
                            if parts[1].startswith('0x'):
                                address = int(parts[1], 16)
                            else:
                                address = int(parts[1])
                            print(f"INFO: Endereço de carregamento alterado para: {address} (0x{address:X})")
                        except ValueError:
                            print(f"AVISO: Diretiva 'address' ignorada (valor inválido): '{line}'")
                    else:
                        print(f"AVISO: Diretiva 'address' ignorada (formato inválido): '{line}'")
                    continue

                # Assume que a linha é uma instrução em BINÁRIO (32 bits)
                try:
                    # Remove espaços em branco e verifica se tem 32 caracteres
                    instruction_bin = line.replace(' ', '')
                    if len(instruction_bin) != 32:
                        raise ValueError("Instrução não tem 32 bits.")
                        
                    instruction = int(instruction_bin, 2) # Lê em BINÁRIO
                    write_memory(address, instruction)
                    address += 1
                except ValueError as e:
                    print(f"AVISO: Linha ignorada (formato inválido): '{line}' - {e}")
                except IndexError:
                    print(f"ERRO: Fim da memória atingido ao carregar o programa.")
                    return False
    except FileNotFoundError:
        print(f"ERRO: Arquivo de programa não encontrado: '{filepath}'")
        return False
    return True

# --- Ponto de Entrada ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python simulator.py <arquivo_de_programa.hex>")
        sys.exit(1)

    program_file = sys.argv[1]

    if load_program_from_file(program_file):
        run_simulation()
