import os
import re
import subprocess
import argparse

def call_llm(prompt):
    """
    TODO: IMPLEMENTA TU API AQUÍ (OpenAI, Anthropic Claude, o local)
    
    # EJEMPLO CON OPENAI:
    # import openai
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o",
    #     messages=[{"role": "system", "content": "Solo devuelve código python en bloques markdown. Nada más."},
    #               {"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    
    # EJEMPLO CON ANTHROPIC:
    # import anthropic
    # client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # response = client.messages.create(
    #     model="claude-3-5-sonnet-20240620",
    #     max_tokens=1000,
    #     system="Solo devuelve código python en bloques markdown. Nada más.",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.content[0].text
    """
    raise NotImplementedError("Debes conectar tu API preferida en la función call_llm() dentro de agent_loop.py")

def extract_python_code(llm_response):
    """ Busca un bloque de código Python puro en la respuesta del LLM. """
    match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
    if match:
        return match.group(1)
    
    # Si omitió la palabra 'python' pero usó markdown:
    match_fallback = re.search(r"```\n(.*?)\n```", llm_response, re.DOTALL)
    if match_fallback:
        return match_fallback.group(1)
        
    return llm_response # En el peor de los casos asume que devolvió el código desnudo

def run_agent_loop(dataset="toy", target_r2=0.999, max_iterations=20):
    print(f"--- Iniciando Bucle Autoresearch (Max Iteraciones: {max_iterations}) ---")
    
    with open("program.md", "r") as f:
        program_instructions = f.read()

    last_output = "No hay corrida previa. Esta es tu primera iteración."
    
    for i in range(1, max_iterations + 1):
        print(f"\n[Iteración {i}] Pensando...")
        
        with open("train.py", "r") as f:
            current_code = f.read()
            
        prompt = f"""
Eres un agente científico autónomo. Tu objetivo principal está aquí:
{program_instructions}

Aquí está tu código actual (train.py):
```python
{current_code}
```

Al ejecutar este código, esta fue la salida de la terminal (última corrida):
```text
{last_output}
```

TAREA: Analiza el resultado. Si la métrica descubierta no es perfecta (R2 < {target_r2}), modifica `train.py`.
Puedes alterar PARAMETROS (EPISODES, BEAM_SIZE) o establecer un nuevo SEED_EXPR para escapar del ruido local, forzando la exploración matemática.
DEVUELVE ÚNICAMENTE EL NUEVO CÓDIGO PYTHON COMPLETO DENTRO DE UN BLOQUE ```python ... ```. No escribas nada más, tu respuesta sobrescribirá el archivo literalmente y se ejecutará de inmediato.
"""
        
        try:
            # 1. Llamar a la red neuronal artificial (LLM)
            llm_response = call_llm(prompt)
            new_code = extract_python_code(llm_response)
            
            # Protección básica por si alucinó e intentó borrar el archivo
            if "import subprocess" not in new_code:
                raise ValueError("El agente devolvió un archivo sospechoso/vacío. Abortando escritura.")
            
            # 2. Aplicar el parche
            with open("train.py", "w") as f:
                f.write(new_code)
                
            print(f"[Iteración {i}] Código actualizado. Ejecutando motores ND2...")
            
            # 3. Ejecutar Train
            result = subprocess.run(["python3", "train.py", "--dataset", dataset], 
                                    capture_output=True, text=True)
            
            # Evita que el contexto se sature con logs ultra largos de MCTS resumiendo las últimas 30 líneas
            stdout_lines = result.stdout.split('\n')
            if len(stdout_lines) > 50:
                short_out = "\n".join(stdout_lines[:10] + ["... [snip] ..."] + stdout_lines[-30:])
            else:
                short_out = result.stdout
                
            last_output = short_out + "\nERRORES:\n" + result.stderr if result.stderr else short_out
            
            # Guardar histórico humano
            with open("experimentos.log", "a") as f:
                f.write(f"\n\n{'='*20} ITERACION {i} {'='*20}\n")
                f.write(last_output)
            
            # 4. Evaluación de Parada
            if "R2: 1.000" in last_output or "R2: 0.999" in last_output or "R2: 0.998" in last_output:
                print(f"\n[!] ¡Métrica cercana a la perfección alcanzada en iteración {i}! Autoresearch completado.")
                break
                
        except Exception as e:
            msg = f"Error crítico de orquestación en Iteración {i}: {str(e)}"
            print(msg)
            last_output = msg # Retroalimenta al LLM con su propio error para que lo intente corregir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "harmonic", "legendre"])
    args = parser.parse_args()
    
    try:
        run_agent_loop(dataset=args.dataset)
    except NotImplementedError as e:
        print(f"\n[X] Falta Configuración: {e}")
        print("Abre 'agent_loop.py' y descomenta/modifica el bloque de OpenAI o Anthropic.\n")
