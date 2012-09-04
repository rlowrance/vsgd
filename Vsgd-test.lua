-- Vsgdd-test.lua
-- unit test of Vsgd.lua

require 'checkGradient'
require 'Vsgd'

test = {}
tester = torch.Tester()

-- assert norm(expected - actual) is small
local function assertEqTensor(actual, expected1, expected2, tol)
   local trace = false
   local tol = tol or 0.1
   local expected = torch.Tensor(2)
   expected[1] = expected1
   expected[2] = expected2
   if trace then
      print('expected', expected)
      print('actual', actual)
      print('expected-actual', expected - actual)
      print('norm diff', torch.norm(expected - actual))
   end
   local diff = torch.norm(expected - actual)
   tester:assertle(diff, tol, 
                   'expected-actual=' .. tostring(expected-actual))
   if false and not (diff <= tol) then
      error('stopping, since assertion failed; diff = ' .. diff .. 
            ' tol = ' .. tol)
   end
end

-- function is from Heath, Scientific Computing, p. 282
-- f(x1, x2) = 0.5 x_1^2 + 2.5 x_2^2
-- the minizer is at the origin [0 0]^T
do
   -- must hide the definition of Heath for the non-class code
   local trace = false
   local Heath = torch.class('Heath')

   function Heath:__init()
      self._nDimensions = 2
   end

   function Heath:nDimensions()
      return self._nDimensions
   end

   function Heath:nSamples()
      return self._nSamples
   end

   -- return fx, df/dx, and df^2/dx^2 at theta and a sample
   -- however, there are no samples in the example
   function Heath:run(theta)
      assert(theta, 'no argument to Heath:run')
      if trace then
         print('\nHeath:run self', self)
      end

      local x1 = theta[1]
      local x2 = theta[2]

      local fx = 0.5 * x1 * x1 + 2.5 * x2 * x2

      local g = torch.Tensor(2)
      g[1] = x1
      g[2] = 5 * x2

      local h = torch.Tensor(2)
      h[1] = 1
      h[2] = 5

      return fx, g, h
   end
end

function test.Heath() -- test Heath class
   local heath = Heath()
   local function f(theta)
      assert(theta)
      return heath:run(theta)
   end

   local function check(expectedFx, 
                        expectedG1, expectedG2,
                        theta)
      local trace = false
      local actualFx, actualG, actualH = f(theta)
      if trace then 
         print('check fx, g, h', actualFx, actualG, actualH)
      end
      tester:asserteq(expectedFx, actualFx, 'fx')
      assertEqTensor(actualG, expectedG1, expectedG2, 0)
      assertEqTensor(actualH, 1, 5, 0)
   end

   check(3, 1, 5, torch.Tensor(2):fill(1))
   check(12, 2, 10, torch.Tensor(2):fill(2))
   check(0, 0, 0, torch.Tensor(2):fill(0))
end


-- function from Heath with specific epsilon and c values
-- results were computed by hand for one iteration
function test.heathTestKnownResults()
   local trace = false
   if trace then print('\n') end

   local heath = Heath()
   if trace then print('Heath instance', heath) end
   local d = heath:nDimensions()

   local vsgd = Vsgd()

   state = {}
   state.epsilon = 1e-10
   state.n0 = 2
   state.c = 10

   -- initial starting point
   local theta = torch.Tensor(d):fill(1)
   if trace then print('state before call') print(state) end
   local function f(theta)
      assert(theta, 'no argument to f')
      return heath:run(theta)
   end
   local theta, seq = vsgd:ld(f, theta, state)
   if trace then print('state after call', state) end

   assertEqTensor(state.g, 1, 5)
   assertEqTensor(state.v,  5.5, 137.5)
   assertEqTensor(state.h,  5.5, 27.5)
   assertEqTensor(state.eta, 0.0331,   0.0066)
   assertEqTensor(state.tau, 2.6364,   2.6364)
   assertEqTensor(theta,     0.9669,   0.9670)
   tester:asserteq(1, #seq, '1 element')
   tester:asserteq(3, seq[1], 'value should be 3')
end

-- attempt to find its minimizer of the function from Heath
function test.heathMinimizer()
   local trace = true
   if trace then print('\n') end

   local heath = Heath()
   local function f(theta)
      assert(theta, 'f is missing theta')
      return heath:run(theta)
   end

   local function minimize(startX1, startX2, n0, steps)
      local trace = false
      assert(startX1)
      assert(startX2)

      state = {}
      state.epsilon = 1e-10 
      state.nSamples = heath:nSamples()
      if n0 == nil then
         state.n0 = state.nSamples
      else
         state.n0 = n0
      end
      state.c = 2
      
      -- attempt to minimize
      local d = heath:nDimensions()
      local theta = torch.Tensor(2)
      theta[1] = startX1
      theta[2] = startX2
      if trace then
         print(string.format('\n%d theta [%f, %f]', 0, theta[1], theta[2]))
      end
      local vsgd = Vsgd()
      steps = steps or 10
      for i = 1, steps do
         theta, seq = vsgd:ld(f, theta, state)
         if trace then
            print(string.format('%d theta [%f, %f] f(theta)= %f', 
                                i, theta[1], theta[2], seq[1]))
         end
      end
      assertEqTensor(theta, 0, 0, .01)
      tester:assertlt(seq[1], 1e-3, 'less than 10^-3')
   end --minimize

   minimize(1, 1, 2, 50) -- same starting point as for known results test
   minimize(0.1, 0.1, 2, 50)
   minimize(100, -100, 2, 200 ) -- as in Heath p. 277 using gradient descent
end

--------------------------------------------------------------------------------
-- Rosenbrock function
--------------------------------------------------------------------------------

do
   local function rosenbrock(theta)
      local x = theta[1]
      local y = theta[2]
      
      local term1 = 1 - x
      local term2 = y - x * x
      
      return term1 * term1 + 100 * term2 * term2
   end

   function rosenbrockOpfunc3(theta)

      local x = theta[1]
      local y = theta[2]

      local f = rosenbrock(theta)

      local d = 2
      local gradient = torch.Tensor(d)
      local a =  2 * (1 - x) * (-1)
      local b = 200 * (y - x * x) * (- 2 * x)
      gradient[1] = 2 * (1 - x) * (-1) + 200 * (y - x * x) * (- 2 * x)
      gradient[2] = 200 * (y - x * x)

      local hessdiag = torch.Tensor(d)
      hessdiag[1] = 1 - 400 * y + 1200 * x * x
      hessdiag[2] = 200 

      return f, gradient, hessdiag
   end
end

function test.rosenbrock()
   local trace = false

   -- check the gradient at a random point
   local point = torch.rand(2)
   local tolerance = 1e-6
   local verbose = trace
   local d, dy, dh = checkGradient(rosenbrockOpfunc3, point, tolerance, verbose)
   if trace then
      print('check gradient results')
      print(' d', d)
      print(' gradient from op func', dy)
      print(' computed gradient for pertubation', dh)
   end
   tester:assertlt(math.abs(d), tolerance, 'norm of dy - dh')



   local function check(theta, expectedF, expectedG, expectedHd)
      local actualF, actualG, actualHd = rosenbrockOpfunc3(theta)
      local tol = 1e-5
      if trace then
         print('theta') print(theta)
         print('expectedF') print(expectedF)
         print('actualF') print(actualF)
         print('expectedG') print(expectedG)
         print('actualG') print(actualG)
         print('expectedHd') print(expectedHd)
         print('actualHd') print(actualHd)
      end

      tester:assertlt(math.abs(actualF - expectedF), tol, 'f')
      tester:assertlt(torch.norm(actualG - expectedG), tol, 'g')
      tester:assertlt(torch.norm(actualHd - expectedHd), tol, 'hd')
   end

   local function makeTensor(x, y)
      local result = torch.Tensor(2)
      result[1] = x
      result[2] = y
      return result
   end

   check(makeTensor(0,0),
         1,
         makeTensor(-2,0),
         makeTensor(1, 200))
   check(makeTensor(1,1),
         0,
         makeTensor(0,0),
         makeTensor(801,200))
   check(makeTensor(2,3),
         101,
         makeTensor(802,-200),
         makeTensor(3601,200))

end
   

      

-- function from Rosenbrock, attempt to find its minimizer, which is [1 1]
function test.rosenbrockMinimizer()
   local trace = true
   if trace then print('\n') end
   
   state = {}
   state.epsilon = 1e-10 -- should not play a role
   state.n0 = 1
   state.c = 20

   -- attempt to minimize
   local d = 2
   -- Cannot start from the minimizer!
   --local theta = torch.Tensor(d):fill(1)
   local theta = torch.Tensor(d):fill(0)
   vsgd = Vsgd()
   for i = 1, 25 do
      theta, seq = vsgd:ld(rosenbrockOpfunc3, theta, state)
      if theta[1] ~= theta[1] or theta[2] ~= theta[2] then
         print('theta', theta)
         error('theta is NaN')
      end
      if trace then
         print('theta', theta)
         print('seq', seq)
         print('state', state)
      end
      print(string.format('%d theta [%f, %f] f(theta) = %f', 
                          i, theta[1], theta[2]. seq[1]))
   end
   assertEqTensor(theta, 1, 1)
end

--------------------------------------------------------------------------------
-- main
--------------------------------------------------------------------------------

if true then
   --tester:add(test.Heath, 'test.Heath')
   --tester:add(test.heathTestKnownResults, 'test.heathTestKnownResults')
   --tester:add(test.heathMinimizer, 'test.heathMinimizer')
   --tester:add(test.rosenbrock, 'test.rosenbrock')
   tester:add(test.rosenbrockMinimizer, 'test.rosenbrockMinimizer')
else
   test:add(test)
end
tester:run()

