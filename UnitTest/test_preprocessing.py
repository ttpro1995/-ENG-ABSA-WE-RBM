import pytest
from Preprocessing.parse_raw_data import strip_between, strip_tag

class TestPreprocessing:
    def test_strip(self):
        start = '>'
        end = '</'
        s1 = '<1>I want this</1>'
        s2 =  '<123>I want this</123>'
        r1 = strip_between(s1, start, end)
        r2 = strip_between(s2)
        expect = 'I want this'
        assert r1 == expect
        assert r2 == expect
        s = u'<2> the food is a melding of moroccan comfort food and spanish tapas fare : tagines , stews and salads , with surprises like baby eggplants and olives where you might not expect them . 2>'
        r = strip_tag(s)
        assert r == ' the food is a melding of moroccan comfort food and spanish tapas fare : tagines , stews and salads , with surprises like baby eggplants and olives where you might not expect them . '