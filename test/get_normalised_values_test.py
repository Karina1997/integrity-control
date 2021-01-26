import unittest
from reformat_data import get_normalised_values


class MyTestCase(unittest.TestCase):
    def test_add_zeros_to_the_middle(self):
        current_state_1 = [134, 128, 31, 62, 6, 0, 144, 32, 8, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 224, 0, 0, 0, 0,
                           0,
                           0, 0, 0, 0, 0, 0, 134, 128, 175, 162, 6, 4, 144, 2, 0, 48, 3, 12, 0, 0, 128, 0, 4, 0, 17,
                           247,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134,
                           0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 11, 1, 0, 0]

        current_state_2 = [134, 128, 31, 62, 6, 0, 144, 32, 8, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 224, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0,
                           134, 128, 145, 62, 7, 4, 16, 0, 0, 0, 0, 3, 16, 0, 0, 0, 4, 0, 0, 246, 0, 0, 0, 0, 12, 0, 0,
                           224, 0, 0, 0, 0, 1, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 64, 0,
                           0, 0, 0, 0, 0, 0, 11, 1, 0, 0,
                           134, 128, 175, 162, 6, 4, 144, 2, 0, 48, 3, 12, 0, 0, 128, 0, 4, 0, 17, 247, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 112,
                           0, 0, 0, 0, 0, 0, 0, 11, 1, 0, 0]

        current_state_3 = [134, 128, 31, 62, 6, 0, 144, 32, 8, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 224, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           134, 128, 145, 62, 7, 4, 16, 0, 0, 0, 0, 3, 16, 0, 0, 0, 4, 0, 0, 246, 0, 0, 0, 0, 12, 0,
                           0, 224, 0, 0, 0, 0, 1, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0,
                           64, 0, 0, 0, 0, 0, 0, 0, 11, 1, 0, 0,
                           134, 128, 175, 162, 6, 4, 144, 2, 0, 48, 3, 12, 0, 0, 128, 0, 4, 0, 17, 247, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0,
                           112, 0, 0, 0, 0, 0, 0, 0, 11, 1, 0, 0]

        expected_res = [128, 134, 128, 31, 62, 6, 0, 144, 32, 8, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 224, 0, 0, 0, 0,
                           0,
                           0, 0, 0, 0, 0, 0, 134, 128, 175, 162, 6, 4, 144, 2, 0, 48, 3, 12, 0, 0, 128, 0, 4, 0, 17,
                           247,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134,
                           0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 11, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        ]

        res, max_len = get_normalised_values([current_state_1, current_state_2, current_state_3])
        self.assertEqual(max_len, 192)
        self.assertEqual(expected_res, res[0])
        self.assertEqual(current_state_2, res[1][1:])
        self.assertEqual(current_state_3, res[2][1:])
        self.assertEqual(res[1][0], 192)
        self.assertEqual(res[2][0], 192)

    def test_cut_values(self):
        current_state_2 = [134, 128, 31, 62, 6, 0, 144, 32, 8, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 224, 0, 0,
                               0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               134, 128, 145, 62, 7, 4, 16, 0, 0, 0, 0, 3, 16, 0, 0, 0, 4, 0, 0, 246, 0, 0, 0, 0, 12, 0,
                               0,
                               224, 0, 0, 0, 0, 1, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 64,
                               0,
                               0, 0, 0, 0, 0, 0, 11, 1, 0, 0,
                               134, 128, 175, 162, 6, 4, 144, 2, 0, 48, 3, 12, 0, 0, 128, 0, 4, 0, 17, 247, 0, 0, 0, 0,
                               0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0,
                               112,
                               0, 0, 0, 0, 0, 0, 0, 11, 1, 0, 0]


        expected_res = [192, 134, 128, 31, 62, 6, 0, 144, 32, 8, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 224, 0, 0,
                               0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               134, 128, 145, 62, 7, 4, 16, 0, 0, 0, 0, 3, 16, 0, 0, 0, 4, 0, 0, 246, 0, 0, 0, 0, 12, 0,
                               0,
                               224, 0, 0, 0, 0, 1, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 16, 148, 134, 0, 0, 0, 0, 64,
                               0,
                               0, 0, 0, 0, 0, 0, 11, 1, 0, 0
                            ]

        res, max_len = get_normalised_values([current_state_2], 128)
        self.assertEqual(max_len, 128)
        self.assertEqual(expected_res, res[0])




if __name__ == '__main__':
    unittest.main()
