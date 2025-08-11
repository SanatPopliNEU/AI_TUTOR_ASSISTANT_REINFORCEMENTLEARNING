"""
Enhanced Question Generation System for RL Tutorial

This module provides properly formatted questions that match their types
(multiple choice, true/false, open-ended) with appropriate content.
"""

import random
from typing import Dict, List, Tuple
from environment.tutoring_environment import QuestionType, DifficultyLevel

class EnhancedQuestionGenerator:
    """Generates properly formatted questions for the RL tutorial system."""
    
    def __init__(self):
        self.question_bank = self._build_question_bank()
    
    def _build_question_bank(self) -> Dict:
        """Build a comprehensive question bank with proper formatting."""
        return {
            'mathematics': {
                DifficultyLevel.EASY: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'What is 15 + 27?',
                            'options': ['42', '35', '41', '44'],
                            'correct': '42'
                        },
                        {
                            'question': 'If you have 8 apples and give away 3, how many do you have left?',
                            'options': ['5', '6', '4', '7'],
                            'correct': '5'
                        },
                        {
                            'question': 'What is 6 × 4?',
                            'options': ['24', '20', '26', '22'],
                            'correct': '24'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'The result of 12 ÷ 3 is 4.',
                            'correct': 'True'
                        },
                        {
                            'question': '5 × 7 equals 30.',
                            'correct': 'False'
                        },
                        {
                            'question': 'Half of 10 is 5.',
                            'correct': 'True'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'What is 20 + 15?',
                            'correct': '35'
                        },
                        {
                            'question': 'Calculate 100 - 25.',
                            'correct': '75'
                        }
                    ]
                },
                DifficultyLevel.MEDIUM: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'What is the area of a rectangle with length 8 and width 6?',
                            'options': ['48', '42', '56', '36'],
                            'correct': '48'
                        },
                        {
                            'question': 'If f(x) = 2x + 3, what is f(5)?',
                            'options': ['13', '11', '15', '10'],
                            'correct': '13'
                        },
                        {
                            'question': 'What is 25% of 80?',
                            'options': ['20', '25', '15', '30'],
                            'correct': '20'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'The volume of a cube with side length 4 is 64.',
                            'correct': 'True'
                        },
                        {
                            'question': 'The circumference of a circle with radius 5 is approximately 31.4.',
                            'correct': 'True'
                        },
                        {
                            'question': 'The square root of 36 is 7.',
                            'correct': 'False'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'Solve for x: 2x + 5 = 17',
                            'correct': '6'
                        },
                        {
                            'question': 'What is the perimeter of a square with side 6?',
                            'correct': '24'
                        }
                    ]
                },
                DifficultyLevel.HARD: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'What is the derivative of f(x) = x³ + 2x² - 5x + 1?',
                            'options': ['3x² + 4x - 5', '3x² + 4x + 5', 'x² + 4x - 5', '3x² - 4x - 5'],
                            'correct': '3x² + 4x - 5'
                        },
                        {
                            'question': 'Which equation represents a parabola opening downward?',
                            'options': ['y = 2x²', 'y = -x² + 3', 'y = x² - 1', 'y = 3x² + 2'],
                            'correct': 'y = -x² + 3'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'The integral of sin(x) dx is -cos(x) + C.',
                            'correct': 'True'
                        },
                        {
                            'question': 'The limit of (sin(x))/x as x approaches 0 is 0.',
                            'correct': 'False'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'Factor completely: x² - 4x + 3',
                            'correct': '(x-1)(x-3)'
                        }
                    ]
                }
            },
            'science': {
                DifficultyLevel.EASY: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'What gas do plants absorb from the atmosphere during photosynthesis?',
                            'options': ['Oxygen', 'Carbon Dioxide', 'Nitrogen', 'Hydrogen'],
                            'correct': 'Carbon Dioxide'
                        },
                        {
                            'question': 'Which planet is closest to the Sun?',
                            'options': ['Venus', 'Earth', 'Mercury', 'Mars'],
                            'correct': 'Mercury'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'Water boils at 100 degrees Celsius at sea level.',
                            'correct': 'True'
                        },
                        {
                            'question': 'The human body has 5 senses.',
                            'correct': 'True'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'Name the three states of matter.',
                            'correct': 'solid liquid gas'
                        }
                    ]
                },
                DifficultyLevel.MEDIUM: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'What is the powerhouse of the cell?',
                            'options': ['Nucleus', 'Mitochondria', 'Ribosome', 'Chloroplast'],
                            'correct': 'Mitochondria'
                        },
                        {
                            'question': 'Which force keeps planets in orbit around the Sun?',
                            'options': ['Magnetic force', 'Nuclear force', 'Gravitational force', 'Electric force'],
                            'correct': 'Gravitational force'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'There are four fundamental forces in nature: gravitational, electromagnetic, strong nuclear, and weak nuclear.',
                            'correct': 'True'
                        },
                        {
                            'question': 'DNA stands for Deoxyribonucleic Acid.',
                            'correct': 'True'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'What is the chemical symbol for gold?',
                            'correct': 'Au'
                        }
                    ]
                },
                DifficultyLevel.HARD: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'Which process describes the conversion of glucose and oxygen into energy, carbon dioxide, and water?',
                            'options': ['Photosynthesis', 'Cellular respiration', 'Fermentation', 'Glycolysis'],
                            'correct': 'Cellular respiration'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'Entropy always increases in an isolated system according to the second law of thermodynamics.',
                            'correct': 'True'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'What is the name of the process by which cells divide to produce gametes?',
                            'correct': 'meiosis'
                        }
                    ]
                }
            },
            'programming': {
                DifficultyLevel.EASY: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'Which of the following is used to print output in Python?',
                            'options': ['print()', 'echo()', 'output()', 'display()'],
                            'correct': 'print()'
                        },
                        {
                            'question': 'What symbol is used for comments in Python?',
                            'options': ['//', '#', '/*', '<!-- -->'],
                            'correct': '#'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'Python is a case-sensitive programming language.',
                            'correct': 'True'
                        },
                        {
                            'question': 'Variables in Python must be declared with a specific type.',
                            'correct': 'False'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'What keyword is used to define a function in Python?',
                            'correct': 'def'
                        }
                    ]
                },
                DifficultyLevel.MEDIUM: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'Which data structure uses LIFO (Last In, First Out)?',
                            'options': ['Queue', 'Stack', 'List', 'Dictionary'],
                            'correct': 'Stack'
                        },
                        {
                            'question': 'What is the time complexity of binary search?',
                            'options': ['O(n)', 'O(log n)', 'O(n²)', 'O(1)'],
                            'correct': 'O(log n)'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'In Python, lists are mutable while tuples are immutable.',
                            'correct': 'True'
                        },
                        {
                            'question': 'Recursion always uses less memory than iteration.',
                            'correct': 'False'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'What method is used to add an element to the end of a list in Python?',
                            'correct': 'append'
                        }
                    ]
                },
                DifficultyLevel.HARD: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'Which design pattern ensures a class has only one instance?',
                            'options': ['Factory', 'Singleton', 'Observer', 'Strategy'],
                            'correct': 'Singleton'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'Dynamic programming is used to solve problems with overlapping subproblems and optimal substructure.',
                            'correct': 'True'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'What is the space complexity of merge sort?',
                            'correct': 'O(n)'
                        }
                    ]
                }
            },
            'language': {
                DifficultyLevel.EASY: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'Which word is a noun in this sentence: "The cat runs quickly"?',
                            'options': ['cat', 'runs', 'quickly', 'the'],
                            'correct': 'cat'
                        },
                        {
                            'question': 'What is the plural of "child"?',
                            'options': ['childs', 'children', 'childes', 'child'],
                            'correct': 'children'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'A verb is an action word.',
                            'correct': 'True'
                        },
                        {
                            'question': 'Adjectives describe nouns.',
                            'correct': 'True'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'What is the past tense of "run"?',
                            'correct': 'ran'
                        }
                    ]
                },
                DifficultyLevel.MEDIUM: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'Which sentence uses correct parallel structure?',
                            'options': [
                                'She likes running, swimming, and to bike.',
                                'She likes running, swimming, and biking.',
                                'She likes to run, swimming, and biking.',
                                'She likes run, swim, and bike.'
                            ],
                            'correct': 'She likes running, swimming, and biking.'
                        },
                        {
                            'question': 'What type of clause is "because it was raining" in the sentence "We stayed inside because it was raining"?',
                            'options': ['Independent clause', 'Dependent clause', 'Noun clause', 'Relative clause'],
                            'correct': 'Dependent clause'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'A dangling modifier is a word or phrase that modifies a word not clearly stated in the sentence.',
                            'correct': 'True'
                        },
                        {
                            'question': 'Passive voice should never be used in writing.',
                            'correct': 'False'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'What is the subject in this sentence: "The quick brown fox jumps over the lazy dog"?',
                            'correct': 'fox'
                        }
                    ]
                },
                DifficultyLevel.HARD: {
                    QuestionType.MULTIPLE_CHOICE: [
                        {
                            'question': 'Which literary device is used in "The wind whispered through the trees"?',
                            'options': ['Metaphor', 'Simile', 'Personification', 'Alliteration'],
                            'correct': 'Personification'
                        }
                    ],
                    QuestionType.TRUE_FALSE: [
                        {
                            'question': 'A subjunctive mood expresses hypothetical or contrary-to-fact situations.',
                            'correct': 'True'
                        }
                    ],
                    QuestionType.SHORT_ANSWER: [
                        {
                            'question': 'Name the rhetorical device that uses repetition of initial consonant sounds.',
                            'correct': 'alliteration'
                        }
                    ]
                }
            }
        }
    
    def get_question(self, topic: str, difficulty: DifficultyLevel, question_type: QuestionType) -> Dict:
        """Get a properly formatted question of the specified type."""
        try:
            questions = self.question_bank[topic][difficulty][question_type]
            return random.choice(questions)
        except KeyError:
            # Fallback to a simple question if specific combination not found
            return {
                'question': f'What do you know about {topic}?',
                'correct': 'Various answers possible',
                'options': ['A lot', 'Some', 'A little', 'Nothing'] if question_type == QuestionType.MULTIPLE_CHOICE else None
            }
