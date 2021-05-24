import spacy
import neuralcoref
import contractions
from spacy import displacy
import re


class EntityParser():

    def __init__(self, character_name, use_coref=False, nlp=None, debug=False):
        """
        Init the global variables used in most parts
        """
        self.debug_ = debug
        if nlp is None:
            self.nlp = spacy.load('en_core_web_sm')
            neuralcoref.add_to_pipe(self.nlp)
        else:
            self.nlp = nlp
        self.use_coref_ = use_coref

        self.character_name_ = character_name
        self.text_ = ""
        self.full_history_ = ""

        self.last_person_entity_ = ""
        self.all_entities_ = {}
        self.description_ = {}
        self.indexes_ = {"PERSON": 0,
                         "GPE": 0,
                         "ADDEDENTITY": 0}
        self.subject_words_ = {"he", "she", "they", "him", "her", "his"}

    def __preprocess_text(self, text):
        """
        Apply the basic preprocessing tecniques to input text.
        """
        if self.debug_:
            print("DEBUG: __preprocess_text")

        text = contractions.fix(text)
        return text

    def __set_history(self, text):
        """
        Save the text into the current full history.
          - full_history: Full history with pre processing techniques or with coreferences explayed
        """
        if self.debug_:
            print("DEBUG: __set_history")

        if self.use_coref_:
            self.full_history_ = str(
                self.nlp(f"{self.full_history_}{text}. ")._.coref_resolved)
        else:
            self.full_history_ = f"{self.full_history_}{text}. "

    def __replace_self_with_name(self, text):
        """
        Replace every occurrencce of "you" with character name
        """
        if self.debug_:
            print("DEBUG: __replace_self_with_name")

        text = re.sub(r'\b{}\b'.format("you"), self.character_name_, text)
        text = re.sub(r'\b{}\b'.format("You"), self.character_name_, text)
        return text

    def __get_not_added_entities(self, entities):
        """
        Add new entities to all_entities dict if it hasn't been added before.
        """
        if self.debug_:
            print("DEBUG: __check_not_added_entities")

        not_added_entities = {}

        for label, entity in entities.items():
            already_added = False

            for e in self.all_entities_.values():
                if entity in e:
                    already_added = True
            for e in self.description_.values():
                if entity in e:
                    already_added = True
            for e in not_added_entities.values():
                if entity in e:
                    already_added = True

            if not already_added:
                not_added_entities[label] = entity

                if "PERSON" in label:
                    self.last_person_entity_ = entity
            else:
                try:
                    self.indexes_["".join(label.split("_")[0])] -= 1
                except:
                    err = "".join(label.split("_")[0])
                    print(f"Error in label {err}")

        return not_added_entities

    def __get_entity_full_name(self, word, type="NOUN", childrens=False, empty=True, full_entity=[]):
        """
        Get the full name of the entity.
        Params:
          word: a NOUN or PROPN spacy token.
        """
        if self.debug_:
            print("DEBUG: __get_entity_full_name")

        if empty:
            full_entity = [str(word)]

        if childrens:
            for children in word.children:
                if type in children.pos_:
                    full_entity.append(str(children))
                    self.__get_entity_full_name(
                        children, type, childrens, False, full_entity)
        else:
            for ancestor in word.ancestors:
                if type in ancestor.pos_:
                    full_entity.append(str(ancestor))
                    self.__get_entity_full_name(
                        ancestor, type, childrens, False, full_entity)

        return full_entity

    def __get_full_entity(self, entity):
        """
        Obtains the full entity name and, if it is a propn, return its description
        """
        if self.debug_:
            print("DEBUG: __get_full_entity")

        full_entity = {}
        propn = ""
        noun = ""
        used_words = []
        for word in self.nlp(entity):
            if str(word) not in used_words:
                if "PROPN" in word.pos_:
                    first = self.__get_entity_full_name(
                        word, "PROPN", childrens=False)
                    second = self.__get_entity_full_name(
                        word, "PROPN", childrens=True)
                    propn = str(word)
                    full_entity[propn] = first if len(
                        first) > len(second) else second
                    used_words.append(propn)
                    used_words.extend(full_entity[propn])

                if "NOUN" in word.pos_:
                    first = self.__get_entity_full_name(
                        word, "NOUN", childrens=False)
                    second = self.__get_entity_full_name(
                        word, "NOUN", childrens=True)
                    noun = str(word)
                    full_entity[noun] = first if len(
                        first) > len(second) else second
                    used_words.append(noun)
                    used_words.extend(full_entity[noun])

        entity_out = ""
        description_out = ""

        if len(propn) > 0:
            entity_out = " ".join(full_entity[propn])
            if len(noun) > 0:
                description_out = " ".join(full_entity[noun])
        else:
            if len(noun) > 0:
                entity_out = " ".join(full_entity[noun])

        return entity_out, description_out

    def __get_full_entities(self, entities):
        """
        Process entities, reducing its form and filling
        the description dict
        """
        if self.debug_:
            print("DEBUG: __get_full_entities")

        ents = {}
        descs = {}
        if len(entities) > 0:
            for label, entity in entities.items():
                ent, desc = self.__get_full_entity(entity)
                ents[label] = ent
                if len(desc) > 0:
                    descs[label] = desc
        return ents, descs

    def __add_entities(self, entities):
        """
        Check if the entity hasn't been added and
        added it if not.
        """
        if self.debug_:
            print("DEBUG: __add_entities")

        not_added_entities, new_descriptions = self.__get_full_entities(
            entities)
        not_added_entities = self.__get_not_added_entities(not_added_entities)
        descriptions = self.__get_descriptions(not_added_entities)
        self.all_entities_.update(not_added_entities)
        self.description_.update(new_descriptions)
        self.description_.update(descriptions)

    def __get_descriptions(self, entities):
        """
        Obtains the apposition modifier of the entity if any
        """
        description = {}
        for label, entity in entities.items():
            for word in self.text_:
                for entity_word in entity.split(" "):
                    if entity_word == str(word):
                        for children in word.children:
                            if "appos" in children.dep_:
                                description[label] = str(children)

        return description

    def __get_entity_label_name(self, label, entity):
        """
        Return and updates the label with their correct index.
        """
        if self.debug_:
            print("DEBUG: __get_entity_label_name")

        try:
            new_label = label
            if "ORG" in new_label:
                new_label = "GPE"
            elif not isinstance(entity, spacy.tokens.span.Span):
                if "dobj" in entity.dep_:
                    new_label = "PERSON"
                elif "pobj" in entity.dep_:
                    new_label = "GPE"

            self.indexes_[new_label] += 1

            if self.debug_:
                print(
                    f"Entity: {entity} /// Label: {label} /// Assigned_label: {new_label}")
            return f"{new_label}_{self.indexes_[new_label] - 1}"
        except Exception as err:
            print(f"Warning: {new_label} not recognized. Error {err}")

    def __get_noun_children(self, word, new_entities):
        """
        Recursively obtains all nouns that are connected 
        with a verb (directly or indirectly)
        """
        if self.debug_:
            print("DEBUG: __get_noun_children")
        for next_word in word.children:
            if "NOUN" in next_word.pos_ or "PROPN" in next_word.pos_:
                new_entities.add(next_word)
                if self.debug_:
                    print(
                        f"Word: {next_word} /// PosTag: {next_word.pos_} /// Dep: {next_word.dep_}")
            else:
                self.__get_noun_children(next_word, new_entities)

    def __replace_subject_word_with_person_entity(self, text):
        """
        Replace every subject word with the last person entity 
        """
        if self.debug_:
            print("DEBUG: __replace_subject_word_with_person_entity")
            print(f"Last person entity: {self.last_person_entity_}")
        for subject in self.subject_words_:
            text = re.sub(r'\b{}\b'.format(subject),
                          self.last_person_entity_, text)

        return text

    def __set_text(self, text):
        """
        Set the text to parse. This function needs to be called
        before parsing.
        Parameters:
          text: text to be parsed.
        """
        if self.debug_:
            print("DEBUG: __set_text")
        text_preprocessed = self.__preprocess_text(text)

        self.__set_history(text_preprocessed)

        if self.use_coref_:
            text_preprocessed = str(self.full_history_.split(". ")[-2])

        # Remove if the sentence has two equals words together
        text_preprocessed = text_preprocessed.split(" ")
        new_text = []
        for i in range(1, len(text_preprocessed)):
            if text_preprocessed[i-1] != text_preprocessed[i]:
                new_text.append(text_preprocessed[i-1])
        new_text.append(text_preprocessed[-1])
        text_preprocessed = " ".join(new_text)

        #text_preprocessed = self.__replace_self_with_name(text_preprocessed)
        self.text_ = self.nlp(text_preprocessed)

        if self.debug_:
            print(f"Text: {self.text_}")

    def __process_entities(self):
        """
        Returns all entities founded using NER by spacy.
        """
        if self.debug_:
            print("DEBUG: __process_entities")
        entities = dict((f"{self.__get_entity_label_name(entity.label_, entity)}", str(entity))
                        for entity in self.text_.ents)

        self.__add_entities(entities)

    def __process_entities_by_verb(self):
        """
        Obtains all new entities (nouns) that are connected
        directly or indirectly to a verb.
        """
        if self.debug_:
            print("DEBUG: __process_entities_by_verb")
        new_entities = set()

        for verb in [word for word in self.text_ if "VERB" in word.pos_]:
            self.__get_noun_children(verb, new_entities)

        new_entities = dict((self.__get_entity_label_name(
            "ADDEDENTITY", entity), str(entity)) for entity in new_entities)

        self.__add_entities(new_entities)

    def naive_coref(self):
        """
        Simplified way of doing coreference, helpfull function
        if you want the output text to be small as possible
        """
        if self.debug_:
            print("DEBUG: naive_coref")

        text = self.__replace_self_with_name(str(self.text_))
        if not self.use_coref_:
            text = self.__replace_subject_word_with_person_entity(text)

        return text

    def display_processed_text(self):
        """
        Display relationship between words in text.
        """
        if self.debug_:
            print("DEBUG: display_processed_text")
        displacy.render(self.text_, jupyter=True, options={
                        'distance': 110, 'arrow_stroke': 2, 'arrow_width': 8})

    def process(self, text, by_verb=True):
        """
        Main function which does the heavy process in order
        to obtain the entities
        """
        if self.debug_:
            print("DEBUG: process")
        self.__set_text(text)
        self.__process_entities()
        if by_verb:
            self.__process_entities_by_verb()
