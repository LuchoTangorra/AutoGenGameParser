from allennlp.predictors.predictor import Predictor
import spacy


class ActionsParser():

    def __init__(self, threshold=0.5, nlp=None, predictor=None, debug=False):
        self.debug_ = debug
        if predictor is None:
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
        else:
            self.predictor = predictor
        if nlp is None:
            self.nlp = spacy.load('en_core_web_sm')
        else:
            self.nlp = nlp
        self.threshold_ = threshold
        self.actions_ = []

    def __get_similarity(self, text1, text2):
        """
        Check how many word both texts have in common.
        """
        if self.debug_:
            print("DEBUG: __get_similarity")

        similarity = 0
        for word1 in text1.split(" "):
            for word2 in text2.split(" "):
                if word1 == word2:
                    similarity += 1

        if self.debug_:
            print(
                f"text1: {text1} vs {text2} /// Similarity score: {similarity}")

        return similarity

    def __fix_entity(self, original_entity, entities, entities_with_descriptions):
        """
        Check similarity between the entity to add
        and the entities we have in order to keep track
        of the entities
        """
        if self.debug_:
            print("DEBUG: __fix_entity")

        similarity_score = []

        for entity in entities_with_descriptions:
            similarity_score.append(
                self.__get_similarity(original_entity, str(entity)))

        max_similarity_score = max(similarity_score)
        if max_similarity_score >= self.threshold_:
            max_index = similarity_score.index(max_similarity_score)
            return entities[max_index]

        return "Deleted"

    def __process_action(self, action, words_list, persons, persons_with_desc, locations, locations_with_desc):
        """
        Process a single action.
        Obtains only arg0 (how) verb (what) arg1 (to) argloc (where).
        """
        if self.debug_:
            print("DEBUG: __process_action")
        action_processed = {}
        arg0 = []
        arg1 = []
        argloc = []
        v = action["verb"]

        for index, tags in enumerate(action["tags"]):
            if "ARG0" in tags:
                arg0.append(words_list[index])
            elif "ARG1" in tags:
                arg1.append(words_list[index])
            elif "ARGM-LOC" in tags or "ARGM-GOL" in tags:
                argloc.append(words_list[index])

        arg0 = self.__fix_entity(" ".join(arg0), persons, persons_with_desc)
        arg1 = self.__fix_entity(" ".join(arg1), persons, persons_with_desc)
        argloc = self.__fix_entity(
            " ".join(argloc), locations, locations_with_desc)

        action_processed[arg0] = [{"V": v}, {
            "RECEIVER": arg1}, {"LOC": argloc}]

        return action_processed

    def __process_actions(self, text, persons, persons_with_desc, locations, locations_with_desc):
        """
        Get all the raw actions from text using allen nlp srl predictor.
        Obtains only arg0 (how) verb (what) arg1 (to) argloc (where).
        """
        if self.debug_:
            print("DEBUG: __process_actions")

        raw_actions = self.predictor.predict(text)

        if self.debug_:
            print(f"raw_actions: {raw_actions}")

        for action in raw_actions["verbs"]:
            action_processed = self.__process_action(
                action, raw_actions["words"], persons, persons_with_desc, locations, locations_with_desc)
            self.actions_.append(action_processed)

            if self.debug_:
                print(f"parsed action: {action_processed}")

    def process(self, text, entities, descriptions):
        """
        Does the heavy processing.
        Params:
          text: sentence to extract the actions
        """
        if self.debug_:
            print("DEBUG: process")
            print(f"Input ents: {entities}")

        persons = [self.nlp(e) for l, e in entities.items() if "PERSON" in l]
        persons_with_desc = []
        for l, e in entities.items():
            if "PERSON" in l:
                if l in descriptions:
                    persons_with_desc.append(
                        self.nlp(f"{e} {descriptions[l]}"))
                else:
                    persons_with_desc.append(self.nlp(e))

        locations = [self.nlp(e) for l, e in entities.items() if "GPE" in l]
        locations_with_desc = []
        for l, e in entities.items():
            if "GPE" in l:
                if l in descriptions:
                    locations_with_desc.append(
                        self.nlp(f"{e} {descriptions[l]}"))
                else:
                    locations_with_desc.append(self.nlp(e))

        if self.debug_:
            print(f"persons ents: {persons}")
            print(f"locations ents: {locations}")

        self.__process_actions(
            text, persons, persons_with_desc, locations, locations_with_desc)
