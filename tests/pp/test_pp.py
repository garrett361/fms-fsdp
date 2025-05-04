from dtest import DTest


class Test(DTest):
    def test(self) -> None:
        print(f"{self.rank=}")
