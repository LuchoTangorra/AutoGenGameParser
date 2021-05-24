import spacy
import neuralcoref_package
from allennlp.predictors.predictor import Predictor

from Parsers.ActionParser import *
from Parsers.PossesionsParser import *
from Parsers.EntityParser import *


class ConfigManager():

    def set_model_hiperparams(self, character_name, threshold, nlp_type, predictor_path):
        """
        Set the model hiperparams.

        Params:
            - character_name: the name of the main character.
            - threshold: value that limits the similary the way ActionsParser keeps track
                        of the same entity with different name (e.g. Dragon / Dragon of Larion).
            - nlp_type: spacy nlp model type. Normally used: en_core_web_sm.
            - predictor_path: allennlp model predictor path. Normally used:
                    https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz
        """
        self._threshold = threshold
        self._nlp = spacy.load(nlp_type)
        neuralcoref.add_to_pipe(self._nlp)

        self.predictor = Predictor.from_path(predictor_path)

        self.character_name = character_name

    def get_entity_parser(self, use_coref=False, debug=False):
        """
        Returns an instance of entity parser.

        Params:
            - use_coref: if use neuralcoref functionality to generate entities.
            - debug: logs. Only use in debug mode.
        """
        return EntityParser(self.character_name, use_coref=use_coref, nlp=self._nlp, debug=debug)

    def get_possesions_parser(self, debug=False):
        """    
        Returns an instance of possesions parser.

        Params:
            - debug: logs. Only use in debug mode.
        """
        return PossesionsParser(nlp=self._nlp, debug=False)

    def get_actions_parser(self, debug=False):
        """    
        Returns an instance of actions parser.

        Params:
            - debug: logs. Only use in debug mode.
        """
        return ActionsParser(nlp=self._nlp, predictor=self.predictor, debug=False)
