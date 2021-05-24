import spacy


class PossesionsParser():

    def __init__(self, nlp=None, debug=False):
        self.debug_ = debug
        if nlp is None:
            self.nlp = spacy.load('en_core_web_sm')
        else:
            self.nlp = nlp
        self.possesions_ = {}
        self.possesives_ = {"have", "has", "wearing"}

    def __get_possesions(self, word, empty=True, entity_possesions=[]):
        """
        Recursively obtains all entity possesions looking if the childrens
        are connected via dobj or conj
        """
        if self.debug_:
            print("DEBUG: __get_possesions")

        if empty:
            entity_possesions = []

        for children in word.children:
            if "conj" in children.dep_ or "dobj" in children.dep_:
                entity_possesions.append(str(children))
                self.__get_possesions(children, False, entity_possesions)

        return entity_possesions

    def __get_entity_possesions(self, text, label, entity):
        """
        Finds the entity in the text and obtains all its possesions.
        """
        if self.debug_:
            print("DEBUG: __get_entity_possesions")

        for word in text:
            for entity_word in entity:
                if str(word) == str(entity_word):
                    if str(word.head) in self.possesives_:
                        self.possesions_[
                            label] = self.__get_possesions(word.head)

    def process(self, text, entities):
        """
        Does the heavy processing.
        Params:
          text: sentence to extract the possesions
          entities: entities in the sentence
        """
        text = self.nlp(text)

        for label, entity in entities.items():
            self.__get_entity_possesions(text, label, self.nlp(entity))
