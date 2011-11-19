from asp.jit import mapreduce_support as mr, asp_module

class ArrayDoublerMRJob(mr.AspMRJob):
    def mapper(self, key, value):
        yield 1, [float(value) * 2]

class ArrayDoubler(object):
    
    def __init__(self):
        self.pure_python = True

    def double_using_template(self, arr):
        import asp.codegen.templating.template as template
        mytemplate = template.Template(filename="templates/double_template.mako", disable_unicode=True)
        rendered = mytemplate.render(num_items=len(arr))

        mod = asp_module.ASPModule()
        # remember, must specify function name when using a string
        mod.add_function("double_in_c", rendered)
        return mod.double_in_c(arr)

    def double_using_scala(self, arr):
        mod = asp_module.ASPModule(use_scala=True)
        # remember, must specify function name when using a string
        rendered = """
        object double_using_scala {
          def double(arr: List[Double]): List[Double] = {
            return arr map { _ * 2.0 }
          }
          def main(args: Array[String]) {
            var arr: List[Double] = scala.util.parsing.json.JSON.parse(args(0)).getOrElse(List()) match {
              case x:List[Double] => x
            }
            var arr2 = double(arr)
            for (s <- arr2) {
              print(s)
              print(" ")
            }
          }
        }
        """
        mod.add_function("double_using_scala", rendered, backend="scala")
        return mod.double_using_scala(arr)

    def double_using_mapreduce(self, arr):
        mod = asp_module.ASPModule(use_mapreduce=True)
        mod.add_mr_function("double_using_mapreduce", ArrayDoublerMRJob)
        return mod.double_using_mapreduce(arr)

    def double(self, arr):
        return map (lambda x: x*2, arr)

# this appears to be necessary because this script will be called as __main__ on
# every worker node.
if __name__ == '__main__':
    ArrayDoublerMRJob().run()
