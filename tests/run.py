import sys
import os.path

sys.path[0] = os.path.dirname(os.getcwd())

from tests import test_patterns, test_parser

# test_patterns.main()
test_parser.main()
