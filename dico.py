with open("/home/jeremie/Documents/liste_francais.txt", "r") as f:
    with open("/home/jeremie/PycharmProjects/NER/dico.txt", "w") as d:
        c = 0
        for line in f:
            d.write(line.encode("utf-8") + str(c)+"\n")
            c += 1
        d.write("unk" + "/t" + str(c))