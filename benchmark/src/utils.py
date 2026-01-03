import re
import signal
from contextlib import contextmanager
from sympy import simplify, sympify, N
from sympy.parsing.latex import parse_latex

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Raise TimeoutException if the block exceeds `seconds` (SIGALRM; main thread only)."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class MathVerifier:
    @staticmethod
    def _extract_boxed_content(text: str) -> str:
        """Extract the last complete \\boxed{...} block using brace balancing."""
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return None

        content_start = idx + 7  # len("\\boxed{")
        balance = 1
        for i in range(content_start, len(text)):
            if text[i] == "{":
                balance += 1
            elif text[i] == "}":
                balance -= 1

            if balance == 0:
                return text[content_start:i]
        return None

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize candidate answers for robust comparison."""
        if s is None:
            return ""
        s = str(s).lower().strip()

        wrappers = [r"\\text\{([^}]+)\}", r"\\mbox\{([^}]+)\}", r"\\mathrm\{([^}]+)\}"]
        for w in wrappers:
            s = re.sub(w, r"\1", s)

        s = s.replace("the answer is", "").replace("answer:", "")
        s = s.replace("\\$", "").replace("$", "")
        s = s.replace("\\%", "%")
        s = s.replace("^{circ}", "").replace("^\\circ", "")

        s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

        s = "".join(s.split())
        s = s.rstrip(".")

        return s

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract a final answer from model output (GSM8K marker, boxed LaTeX, or heuristics)."""
        match = re.search(r"####\s*(.+?)(?:\n|$)", text)
        if match:
            return match.group(1).strip()

        boxed = MathVerifier._extract_boxed_content(text)
        if boxed:
            return boxed.strip()

        patterns = [
            r"[Tt]he final answer is[:\s]+([^\n\.]+)",
            r"[Tt]he answer is[:\s]+([^\n\.]+)",
            r"Answer[:\s]+([^\n\.]+)",
            r"=[:\s]+([^\n\.]+)",
        ]
        for p in patterns:
            matches = re.findall(p, text)
            if matches:
                candidate = matches[-1].strip()
                if len(candidate) < 50:
                    return candidate

        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            last_line = lines[-1]
            if re.match(r"^[\d\.\+\-\*/\^\(\)\{\}\\=a-zA-Z]+$", last_line):
                return last_line

        return ""

    @staticmethod
    def is_equivalent(pred: str, gold: str, tolerance=1e-4) -> bool:
        """Check equivalence via exact, numeric, then symbolic comparison (with timeout)."""
        pred = MathVerifier.normalize_answer(pred)
        gold = MathVerifier.normalize_answer(gold)

        if not pred:
            return False

        if pred == gold:
            return True

        try:
            p_val = float(pred.replace(",", ""))
            g_val = float(gold.replace(",", ""))
            return abs(p_val - g_val) < tolerance
        except:
            pass

        try:
            with time_limit(2):
                try:
                    p_expr = parse_latex(pred)
                    g_expr = parse_latex(gold)
                except:
                    p_expr = sympify(pred, rational=True)
                    g_expr = sympify(gold, rational=True)

                diff = simplify(p_expr - g_expr)
                if diff == 0:
                    return True

                if abs(N(diff)) < tolerance:
                    return True
        except (TimeoutException, Exception):
            pass

        return False