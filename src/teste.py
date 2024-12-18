import pretty_midi
from input_representation import InputRepresentation

# Supondo que o código acima já esteja definido/importado aqui

# Nome do arquivo MIDI a ser processado
nome_arquivo_midi = 'RM-C001.MID'

try:
    # Cria uma instância de InputRepresentation
    input_representation = InputRepresentation(nome_arquivo_midi)

    # Obtém os eventos REMI do arquivo MIDI
    remi_events = input_representation.get_remi_events()

    # Exibe os eventos
    print("Eventos REMI extraídos do arquivo MIDI:")
    for event in remi_events:
        print(event)

except Exception as e:
    print(f"Ocorreu um erro ao processar o arquivo MIDI: {e}")