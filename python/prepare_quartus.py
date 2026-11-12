"""Create clean Quartus projects without distributing generated databases."""
from pathlib import Path
import shutil
from project_config import MODEL
ROOT=Path(__file__).resolve().parents[1]
for kind in ['folded','parallel']:
    d=ROOT/'build/quartus'/kind;d.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'model/weights_rom.mem',d/'weights_rom.mem')
    text=f'''set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CEBA4F23C7
set_global_assignment -name TOP_LEVEL_ENTITY core_{kind}
set_global_assignment -name SYSTEMVERILOG_FILE ../../../rtl/core_{kind}.sv
set_global_assignment -name SEARCH_PATH ../../../rtl
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name SDC_FILE mlp.sdc
set_global_assignment -name SEED 1
set_global_assignment -name NUM_PARALLEL_PROCESSORS 4
set_location_assignment PIN_M9 -to clk
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to clk
'''
    for port in ['rst_n','in_valid','in_ready','out_valid','out_ready','feat_0*','feat_1*','feat_2*','feat_3*','class_id*','confidence*','led_ok','led_warn','led_crit','halt','reserved_unused']:
        text+=f'set_instance_assignment -name VIRTUAL_PIN ON -to {port}\n'
    for k,v in MODEL['SCALES'].items():text+=f'set_parameter -name {k.upper()} {v}\n'
    (d/'mlp.qsf').write_text(text)
    (d/'mlp.qpf').write_text('QUARTUS_VERSION = "20.1"\nPROJECT_REVISION = "mlp"\n')
    (d/'mlp.sdc').write_text('''create_clock -name clk -period 50.000 [get_ports clk]
derive_clock_uncertainty
set_false_path -from [get_ports {rst_n in_valid out_ready feat_0* feat_1* feat_2* feat_3*}]
set_false_path -to [get_ports {in_ready out_valid class_id* confidence* led_ok led_warn led_crit halt reserved_unused}]
''')
    print(d/'mlp.qpf')
