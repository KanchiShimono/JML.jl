using JML
using Base.Test

# write your own tests here

anyerrors = false

testcases = [
            "FunctionsTest.jl"
           ]

Base.info("Running tests...")

for case in testcases
    try
        include(case)
        println("\t\033[1m\033[32mPASSED\033[0m: $(case)")
    catch e
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(case)")
    end
end

if anyerrors
    throw("Test failed")
end
