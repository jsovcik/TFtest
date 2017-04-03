from reader import read_words

tag_lex = {"O":0, "B-C": 1, "B-P":2, "I-C":3, "I-P":4, "E-C":5, "E-P":6, "S-C":7, "S-P":8}

class Tag():

    position_tags = ["I", "O", "B", "E", "S"]
    entity_tags = ["C", "P"]

    def __init__(self, position=None, entity=None):
        self._pos = position
        self._ent = entity
        if (self._pos == "O"):
            self._tag = self._pos
        else:
            self._tag = self._pos+"-"+self._ent

    @property
    def tag(self):
        return self._tag

    @property
    def pos(self):
        return self._pos

    @property
    def ent(self):
        return self._ent

    @tag.setter
    def set(self, position = None, entity = None):
        if (position == "O"):
            self._tag = position
        else:
            self._tag = position+"-"+entity

    @pos.setter
    def set(self, position):
        self._pos = position
        if (position == "O"):
            self._ent = None




class NamedEntity():

    def __init__(self, name, ent_tag):
        self.name = name
        self.tag = ent_tag

def tag_str_list (text, name, entTag):
    """takes list for a text and a list of word with an entity tag and return the list of tags corresponding to the text"""
    tag_list = []
    n = len(text)
    m = len(name)
    i = 0
    while (i< n):
        if (text[i] != name[0]):
            tag_list.append("O")
            i = i+1
        else:
            if (m == 1):
                tag_list.append("S-"+entTag)
                i = i + 1
            else:
                # check if the following in text is the entity given in st_list
                is_ent = (i + m) <= n
                j = 1
                while (is_ent and (j < m)):
                    is_ent = (text[i+j] == name[j])
                    j = j+1
                # update the tag list accordingly
                if is_ent:
                    tag_list.append("B-"+entTag)
                    for k in range(1, m-1):
                        tag_list.append("B-"+entTag)
                    tag_list.append("E-"+entTag)
                    i = i+m
                else:
                    tag_list.append("O")
                    i = i + 1
    return tag_list

def add_tag(txt, tg_list, name, tag):
    """modify the tag list associated to txt with the name entity def by (name, tag)"""
    n = len(txt)
    m = len(name)
    i = 0
    while (i < n):
        if (txt[i] != name[0]):
            i += 1
        else:
            if (m == 1):
                tg_list[i]="S-"+tag
                i += 1
            else:
                # check if the following in text is the entity given in st_list
                is_ent = (i + m) <= n
                j = 1
                while (is_ent and (j < m)):
                    is_ent = (txt[i+j] == name[j])
                    j += 1
                # update the tag list accordingly
                if is_ent:
                    tg_list[i] = "B-"+tag

                    for k in range(1, m-1):
                        tg_list[i+k] = "I-"+tag
                    tg_list[i+m-1] = "E-"+tag
                    i += m
                else:
                    i += 1
    return ()

def print_tag_list (tg_l):
    """test fc"""
    for i in tg_l:
        print(i.ent_tag, i.pos_tag)

def preprocess_tr_data (filename, entList):
    """return a tag list matching the text from filename and the NE from entList"""
    text = read_words (filename)
    ent = entList.pop()
    tg_list = tag_str_list(text, ent.name, ent.tag)
    for ent in entList:
        add_tag(text, tg_list, ent.name, ent.tag)
    for i in range(len(tg_list)):
        tg_list[i] = tag_lex[tg_list[i]]
    return tg_list

#
# elist = [NamedEntity(["open", "source"], "P"), NamedEntity(["alfresco"], "C")]
# print(preprocess_tr_data("/home/jeremie/PycharmProjects/NER/tagtest", elist))