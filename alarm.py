import winsound

def sound_alarm():
    duration = 1000  # Duración en milisegundos
    freq = 440  # Frecuencia en Hz
    winsound.Beep(freq, duration)
