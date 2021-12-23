
from consola import Consola

if __name__=="__main__":
    
    cmd = Consola()
    cmd.prompt = '-------------------------------------------CMD------------------------------------\n------------------------------------ CLASIFICADOR DE -----------------------------------\n---------------------------------------- FRUTAS ----------------------------------\nCMD >>> '
    cmd.cmdloop('Iniciando entrada de comandos...')